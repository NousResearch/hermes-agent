"""Contracts for platform metadata registered in the CLI config surface."""


def test_matrix_metadata_registers_the_runtime_home_target_env_vars():
    from gateway.run import _home_target_env_var, _home_thread_env_var
    from hermes_cli.config import OPTIONAL_ENV_VARS

    home_target = _home_target_env_var("matrix")
    registered_home_vars = {
        name for name in OPTIONAL_ENV_VARS if name.startswith("MATRIX_HOME_")
    }

    assert registered_home_vars == {
        home_target,
        f"{home_target}_NAME",
        _home_thread_env_var("matrix"),
    }
