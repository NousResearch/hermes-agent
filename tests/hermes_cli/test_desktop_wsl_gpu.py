from hermes_cli.main_desktop import _prefer_wsl_d3d12


def test_selects_available_wsl_driver_before_spawning_electron():
    env = {"PATH": "/bin"}
    _prefer_wsl_d3d12(env, available=True)
    assert env == {"PATH": "/bin", "GALLIUM_DRIVER": "d3d12"}


def test_missing_driver_and_explicit_mesa_choices_are_preserved():
    overrides = [
        {"GALLIUM_DRIVER": "llvmpipe"},
        {"GALLIUM_DRIVER": ""},
        {"MESA_LOADER_DRIVER_OVERRIDE": "zink"},
        {"LIBGL_ALWAYS_SOFTWARE": "1"},
        {"LIBGL_DRIVERS_PATH": "/custom/mesa"},
    ]
    for original in overrides:
        env = dict(original)
        _prefer_wsl_d3d12(env, available=True)
        assert env == original
    env = {}
    _prefer_wsl_d3d12(env, available=False)
    assert env == {}
