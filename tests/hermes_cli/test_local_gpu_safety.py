"""GPU identity and failure semantics, not merely a mocked environment dictionary."""
from types import SimpleNamespace

import pytest

from hermes_cli import config as config_mod
from hermes_cli.local_runtime import hardware as hw

UUID0 = 'GPU-00000000-0000-0000-0000-000000000001'
UUID1 = 'GPU-11111111-0000-0000-0000-000000000002'


@pytest.fixture
def cards(monkeypatch):
    rows = [
        dict(index=3, uuid=UUID0, pci_bus_id='00000000:65:00.0', gpu_name='Display',
             total_bytes=6 << 30, free_bytes=3 << 30, used_bytes=3 << 30,
             gpu_pci_id=None, gpu_util_percent=20),
        dict(index=7, uuid=UUID1, pci_bus_id='00000000:01:00.0', gpu_name='Compute',
             total_bytes=8 << 30, free_bytes=8 << 30, used_bytes=0,
             gpu_pci_id=None, gpu_util_percent=0),
    ]
    monkeypatch.setattr(hw, 'nvidia_cards', lambda: rows)
    monkeypatch.setattr(hw, '_configured_gpu_selection', lambda: 'auto')
    for key in ('CUDA_VISIBLE_DEVICES', 'GGML_VK_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES'):
        monkeypatch.delenv(key, raising=False)
    return rows


def test_numeric_yaml_zero_is_not_auto(monkeypatch):
    monkeypatch.setattr(config_mod, 'load_config_readonly', lambda: {'local_runtime': {'gpu': 0}})
    assert hw._configured_gpu_selection() == '0'


def test_cuda_pins_uuid_not_row_or_pci_ordinal(cards, monkeypatch):
    monkeypatch.setattr(hw, '_configured_gpu_selection', lambda: '7')
    assert hw.gpu_pin_env('cuda')['CUDA_VISIBLE_DEVICES'] == UUID1


def test_uuid_selection_survives_enumeration_change(cards, monkeypatch):
    monkeypatch.setattr(hw, '_configured_gpu_selection', lambda: UUID1)
    cards[0]['total_bytes'] = 48 << 30
    assert hw.selected_nvidia_card()['uuid'] == UUID1


def test_auto_budget_and_placement_agree(cards):
    selected = hw.selected_nvidia_card()
    assert hw.gpu_pin_env('cuda')['CUDA_VISIBLE_DEVICES'] == selected['uuid']


def test_missing_explicit_card_never_uses_display_card(cards, monkeypatch):
    monkeypatch.setattr(hw, '_configured_gpu_selection', lambda: '99')
    with pytest.raises(ValueError, match='(?i)(unavailable|not detected|not found)'):
        hw.gpu_pin_env('cuda')


def test_single_gpu_explicit_choice_still_pins(cards, monkeypatch):
    del cards[0]
    monkeypatch.setattr(hw, '_configured_gpu_selection', lambda: '7')
    assert hw.gpu_pin_env('cuda')['CUDA_VISIBLE_DEVICES'] == UUID1


def test_auto_honours_uuid_visibility_prefix(cards, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', UUID0[:37])
    assert hw.selected_nvidia_card()['uuid'] == UUID0


def test_empty_visibility_never_prices_an_invisible_gpu(cards, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    assert hw.selected_nvidia_card() is None
    assert hw.gpu_pin_env('cuda') == {'CUDA_VISIBLE_DEVICES': '-1'}


def test_visibility_restricts_auto_even_for_multicard_list(cards, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', UUID0 + ',-1,' + UUID1)
    assert hw.selected_nvidia_card()['uuid'] == UUID0


def test_nvidia_parser_uses_explicit_indices_and_skips_only_bad_telemetry(monkeypatch):
    monkeypatch.setattr(hw, '_nvidia_smi_path', lambda: 'nvidia-smi')
    monkeypatch.setattr(hw, '_gpu_query_cache', None)
    output = (f'6144, 3072, Display, 0x1F0810DE, 3072, N/A, 3, {UUID0}, 00000000:65:00.0\n'
              f'8192, 8192, Compute, 0x248410DE, 0, 0, 7, {UUID1}, 00000000:01:00.0\n')
    monkeypatch.setattr(hw.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout=output))
    found = hw.nvidia_cards()
    assert [(c['index'], c['uuid']) for c in found] == [(3, UUID0), (7, UUID1)]
    assert found[0]['gpu_util_percent'] is None


def test_inherited_cuda_ordinal_is_not_an_smi_index(cards, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setattr(hw, '_cuda_visible_uuids', lambda: [UUID1])
    # CUDA fastest-first device 0 is the compute card, not smi row/index 0.
    assert hw.selected_nvidia_card()['uuid'] == UUID1
    assert hw.gpu_pin_env('cuda') == {'CUDA_VISIBLE_DEVICES': UUID1}


def test_cuda_allocator_pool_uses_selected_physical_bus(cards, monkeypatch):
    import ctypes
    calls = []
    monkeypatch.setattr(hw, '_configured_gpu_selection', lambda: '7')

    def by_bus(device, bus):
        calls.append(bus)
        device._obj.value = 42
        return 0

    def total(result, device):
        assert device.value == 42
        result._obj.value = 8 << 30
        return 0

    def attribute(result, name, device):
        result._obj.value = 0
        return 0

    cuda = SimpleNamespace(cuInit=lambda _: 0, cuDeviceGetByPCIBusId=by_bus,
                           cuDeviceTotalMem_v2=total, cuDeviceGetAttribute=attribute)
    monkeypatch.setattr(ctypes, 'CDLL', lambda *a, **k: cuda)
    assert hw._cuda_driver_pool() == (8 << 30, False)
    assert calls == [b'00000000:01:00.0']


def test_vulkan_mixed_vendor_uses_engine_device_namespace(monkeypatch):
    from pathlib import Path
    from hermes_cli.local_runtime import devices
    engine = SimpleNamespace(backend='vulkan', binary=Path('test-engine/llama-server'))
    monkeypatch.setattr(hw, '_configured_engine', lambda: engine)
    monkeypatch.setattr(hw, '_configured_gpu_selection', lambda: 'Vulkan1')
    monkeypatch.setattr(hw, '_engine_cards_cache', {})
    monkeypatch.setattr(hw, 'nvidia_cards', lambda: pytest.fail('must not use smi on Vulkan'))
    monkeypatch.setattr(devices, 'probe_devices', lambda *a: [
        dict(name='Vulkan0', description='Intel display', type=2, total=16 << 30, free=12 << 30),
        dict(name='Vulkan1', description='AMD compute', type=1, total=8 << 30, free=8 << 30),
    ])
    assert len(hw.gpu_cards()) == 2
    assert hw.selected_gpu_card()['gpu_name'] == 'AMD compute'
    assert hw.gpu_device_args('vulkan', 'Vulkan1') == ['--device', 'Vulkan1', '--split-mode', 'none']
    monkeypatch.setattr(hw, '_ram_bytes', lambda: (32 << 30, 16 << 30))
    assert hw.probe_budget(planning=True).total_device_bytes == 8 << 30


def test_bootstrap_snapshots_gpu_before_planning_and_spawn(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from hermes_cli.local_runtime import bootstrap, binaries, supervisor
    monkeypatch.setattr(bootstrap, '_SUPERVISOR', None)
    monkeypatch.setattr(hw, 'nvidia_cards', lambda: [dict(index=1, id=UUID1, uuid=UUID1,
        gpu_name='Compute', total_bytes=8 << 30, free_bytes=8 << 30)])
    monkeypatch.setattr(bootstrap, '_SERVING_ENGINE', None)
    monkeypatch.setattr(bootstrap, '_SERVING_GPU', None)
    monkeypatch.setattr(bootstrap, '_cross_process_boot_lock', nullcontext)
    monkeypatch.setattr(bootstrap, 'adopt_legacy_models', list)
    monkeypatch.setattr('hermes_cli.local_runtime.endpoint._state_endpoint', lambda: None)
    monkeypatch.setattr(bootstrap, 'models_dir', lambda: tmp_path / 'models')
    monkeypatch.setattr(bootstrap, 'runtimes_root', lambda: tmp_path)
    monkeypatch.setattr(binaries, 'installed_engine', lambda *_: SimpleNamespace(
        binary=tmp_path / 'llama-server', backend='cuda', tag='b1'))
    monkeypatch.setattr(bootstrap, '_admitted_models_max', lambda *a: 1)
    planned = []

    def presets(*a, **kw):
        planned.append(hw._effective_gpu_selection())
        return tmp_path / 'presets.ini'

    monkeypatch.setattr(bootstrap, '_generate_presets', presets)
    # A different profile preference must not leak into boot pricing.
    monkeypatch.setattr(hw, '_configured_gpu_selection', lambda: '0')
    captured = []

    class Boot:
        def __init__(self, *a, **kw):
            captured.append(kw)
            self.base_url = 'http://test/v1'
        def start(self):
            return None
        def stop(self):
            return None

    monkeypatch.setattr(supervisor, 'LlamaServerSupervisor', Boot)
    monkeypatch.setattr(bootstrap, '_stop_at_exit', lambda: None)
    monkeypatch.setattr(bootstrap, '_start_idle_sweeper', lambda _: None)
    try:
        assert bootstrap.ensure_local_runtime({'local_runtime': {'enabled': True, 'gpu': UUID1}}, force=True)
        assert planned == [UUID1]
        assert captured[0]['gpu'] == UUID1
        assert captured[0]['backend'] == 'cuda'
    finally:
        bootstrap.shutdown_local_runtime()


@pytest.fixture
def recorded_router(cards, tmp_path, monkeypatch):
    from hermes_cli.local_runtime import bootstrap, recovery
    state = {'pid': 123, 'owner_pid': 456, 'backend': 'cuda',
             'executable': str(tmp_path / 'llama-server'), 'engine_tag': 'test'}
    monkeypatch.setattr(bootstrap, '_SERVING_GPU', None)
    monkeypatch.setattr(bootstrap, '_SERVING_ENGINE', None)
    monkeypatch.setattr(recovery, 'read_state', lambda: state)
    monkeypatch.setattr(recovery, 'recorded_process', lambda observed: SimpleNamespace(pid=123))
    monkeypatch.setattr(hw, '_configured_gpu_selection', lambda: UUID1)
    monkeypatch.setattr(hw, '_device_pool_view', lambda: (8 << 30, False))
    monkeypatch.setattr(hw, '_ram_bytes', lambda: (32 << 30, 16 << 30))
    return state


@pytest.mark.parametrize('requested', [UUID0, UUID1])
def test_legacy_serving_gpu_is_unknown_not_the_requester_preference(recorded_router, monkeypatch, requested):
    from hermes_cli.web_routers import local_models
    monkeypatch.setattr(hw, '_configured_gpu_selection', lambda: requested)
    observed = local_models._hardware_payload()
    assert len(observed['gpus']) == 2
    assert observed['gpu_selected'] is None
    assert observed.get('gpu_name') is None
    assert 'running local server' in observed['gpu_error'].lower()
    assert observed['gpu'] == requested  # The saved preference remains distinct from observation.


def test_unknown_serving_gpu_cannot_be_priced_or_pinned(recorded_router):
    from hermes_cli.local_runtime.gpu_selection import GpuSelectionError
    budget = hw.probe_budget(planning=True)
    assert budget.total_device_bytes == 0
    assert budget.usable_vram_bytes == 0
    with pytest.raises(GpuSelectionError, match='running local server'):
        hw.gpu_pin_env('cuda')


def test_reported_foreign_pin_still_overrides_the_requester(recorded_router):
    from hermes_cli.web_routers import local_models
    recorded_router['gpu_selection'] = UUID0
    observed = local_models._hardware_payload()
    assert observed['gpu_selected'] == 3
    assert observed['gpu_name'] == 'Display'
    assert observed['gpu_error'] is None


def test_absent_router_can_still_plan_the_configured_gpu(recorded_router, monkeypatch):
    from hermes_cli.local_runtime import recovery
    from hermes_cli.web_routers import local_models
    monkeypatch.setattr(recovery, 'recorded_process', lambda state: None)
    observed = local_models._hardware_payload()
    assert observed['gpu_selected'] == 7
    assert observed['gpu_name'] == 'Compute'
    assert observed['gpu_error'] is None
