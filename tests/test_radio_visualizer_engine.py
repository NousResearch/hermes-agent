from collections import deque

from radio import level_meter


def test_feature_snapshot_returns_safe_defaults_when_meter_inactive(monkeypatch):
    monkeypatch.setattr(level_meter, '_levels', deque([], maxlen=64))
    monkeypatch.setattr(level_meter, '_running', False)
    monkeypatch.setattr(level_meter, '_process', None)

    snap = level_meter.get_feature_snapshot(16)

    assert snap.levels == [0.0] * 16
    assert snap.energy == 0.0
    assert snap.peak == 0.0
    assert snap.transient == 0.0
    assert snap.motion == 0.0
    assert snap.decay == 0.0
    assert snap.active is False


def test_feature_snapshot_resamples_recent_levels_and_computes_motion(monkeypatch):
    monkeypatch.setattr(level_meter, '_levels', deque([-50.0, -40.0, -20.0, -10.0], maxlen=64))
    monkeypatch.setattr(level_meter, '_running', True)

    class _Proc:
        def poll(self):
            return None

    monkeypatch.setattr(level_meter, '_process', _Proc())

    snap = level_meter.get_feature_snapshot(8)

    assert len(snap.levels) == 8
    assert all(0.0 <= value <= 1.0 for value in snap.levels)
    assert 0.0 <= snap.energy <= 1.0
    assert 0.0 <= snap.peak <= 1.0
    assert snap.motion > 0.0
    assert snap.transient > 0.0
    assert snap.decay >= 0.0
    assert snap.active is True


def test_render_rows_returns_requested_dimensions():
    from radio.visualizer_engine import render_rows

    rows = render_rows(
        preset_name='braille',
        width=16,
        rows=3,
        paused=False,
        position=0.0,
        title_seed='test-track',
    )

    assert len(rows) == 3
    assert all(len(row) == 16 for row in rows)


def test_render_rows_returns_strings_even_when_meter_inactive(monkeypatch):
    from radio import visualizer_engine

    monkeypatch.setattr(
        visualizer_engine,
        'get_feature_snapshot',
        lambda width, smoothing=0.0: level_meter.VisualizerFeatures(
            levels=[0.0] * width,
            energy=0.0,
            peak=0.0,
            transient=0.0,
            motion=0.0,
            decay=0.0,
            active=False,
        ),
    )

    rows = visualizer_engine.render_rows(
        preset_name='braille',
        width=12,
        rows=2,
        paused=False,
        position=4.2,
        title_seed='fallback-seed',
    )

    assert len(rows) == 2
    assert all(isinstance(row, str) for row in rows)
    assert all(len(row) == 12 for row in rows)


def test_visualizer_name_round_trips_through_radio_config(tmp_path, monkeypatch):
    from radio import config as radio_config

    monkeypatch.setattr(radio_config, 'RADIO_DIR', tmp_path)
    monkeypatch.setattr(radio_config, 'CONFIG_PATH', tmp_path / 'config.yaml')

    radio_config.set_visualizer('mirror')

    assert radio_config.get_visualizer() == 'mirror'


def test_build_menu_items_includes_visualizer_presets(monkeypatch):
    from radio import menu

    monkeypatch.setattr(menu, 'list_presets', lambda: ['braille', 'mirror'])

    items = menu.build_menu_items(active_visualizer='mirror')

    visualizer_items = [item for item in items if item.action == 'visualizer']
    assert [item.data['name'] for item in visualizer_items] == ['braille', 'mirror']
    assert visualizer_items[1].sublabel == 'active'


def test_mirror_mode_is_horizontally_symmetric_even_without_mirror_flag(monkeypatch):
    from radio import visualizer_engine

    monkeypatch.setattr(
        visualizer_engine,
        'load_preset',
        lambda name=None: {
            'name': 'mirror',
            'mode': 'mirror',
            'chars': 'ascii',
            'rows': 1,
            'width': 5,
            'attack': 100.0,
            'decay': 100.0,
            'center_boost': 0.0,
            'mirror': False,
        },
    )
    monkeypatch.setattr(
        visualizer_engine,
        'get_feature_snapshot',
        lambda width, smoothing=0.0: level_meter.VisualizerFeatures(
            levels=[0.1, 0.2, 0.7, 1.0, 0.3][:width],
            energy=0.5,
            peak=1.0,
            transient=0.2,
            motion=0.3,
            decay=0.0,
            active=True,
        ),
    )

    row = visualizer_engine.render_rows(
        preset_name='mirror',
        width=5,
        rows=1,
        paused=False,
        position=0.0,
        title_seed='mirror-seed',
    )[0]

    assert row == row[::-1]


def test_scatter_mode_uses_transient_features_instead_of_fallback_bars(monkeypatch):
    from radio import visualizer_engine

    monkeypatch.setattr(
        visualizer_engine,
        'load_preset',
        lambda name=None: {
            'name': 'scatter',
            'mode': 'scatter',
            'chars': 'dots',
            'rows': 3,
            'width': 12,
            'attack': 100.0,
            'decay': 100.0,
            'center_boost': 0.0,
            'mirror': False,
        },
    )
    monkeypatch.setattr(
        visualizer_engine,
        'get_feature_snapshot',
        lambda width, smoothing=0.0: level_meter.VisualizerFeatures(
            levels=[0.0] * width,
            energy=0.2,
            peak=0.3,
            transient=1.0,
            motion=0.5,
            decay=0.0,
            active=True,
        ),
    )
    monkeypatch.setattr(
        visualizer_engine,
        '_synthetic_snapshot',
        lambda width, position, title_seed: level_meter.VisualizerFeatures(
            levels=[0.0] * width,
            energy=0.0,
            peak=0.0,
            transient=0.0,
            motion=0.0,
            decay=0.0,
            active=False,
        ),
    )

    rows = visualizer_engine.render_rows(
        preset_name='scatter',
        width=12,
        rows=3,
        paused=False,
        position=0.0,
        title_seed='scatter-seed',
    )

    assert ''.join(rows).strip()
