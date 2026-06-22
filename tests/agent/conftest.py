"""conftest for agy_cli_client tests.

Registers the ``requires_ls_binary`` mark so pytest doesn't warn about
"unknown mark" when developers run with the default warning config.
"""
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_ls_binary: requires the Antigravity language_server binary",
    )
