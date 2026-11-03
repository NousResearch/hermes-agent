"""The unified runtime's init path must build per-profile registries, not process-wide ones."""

from gateway.hooks import ProfileHookRegistries
from gateway.run_runtime_init import GatewayRuntimeInitMixin


def test_init_registries_builds_per_profile_hook_registries():
    class _Runner(GatewayRuntimeInitMixin):
        def __getattr__(self, name):  # unrelated collaborators the init path touches
            return lambda *a, **k: {}

    runner = _Runner()
    GatewayRuntimeInitMixin._init_registries_and_clocks(runner)
    assert isinstance(runner.__dict__["hooks"], ProfileHookRegistries)
