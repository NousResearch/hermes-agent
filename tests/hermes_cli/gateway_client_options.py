"""Explicit two-checkout native client/runtime compatibility probe option."""


def pytest_addoption(parser):
    parser.addoption("--client-test-runtime-root", default=None,
                     help="Checkout used only for the disposable gateway peer in CLI native tests")
