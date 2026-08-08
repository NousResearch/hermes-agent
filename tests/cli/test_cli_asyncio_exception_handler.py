import asyncio
import inspect

from cli import HermesCLI, _install_cli_asyncio_exception_filter


def test_cli_exception_filter_installs_on_running_loop_and_suppresses_closed_loop():
    delegated = []

    def previous_handler(loop, context):
        delegated.append(context)

    async def scenario():
        loop = asyncio.get_running_loop()
        original_handler = loop.get_exception_handler()
        try:
            loop.set_exception_handler(previous_handler)
            _install_cli_asyncio_exception_filter()

            installed_handler = loop.get_exception_handler()
            assert installed_handler is not None
            assert installed_handler is not previous_handler

            loop.call_exception_handler(
                {"exception": RuntimeError("Event loop is closed")}
            )
            assert delegated == []

            other_context = {"exception": ValueError("real failure")}
            loop.call_exception_handler(other_context)
            assert delegated == [other_context]
        finally:
            loop.set_exception_handler(original_handler)

    asyncio.run(scenario())


def test_cli_installs_exception_filter_after_prompt_toolkit_loop_starts():
    source = inspect.getsource(HermesCLI.run)

    assert "app.run(pre_run=_install_cli_asyncio_exception_filter)" in source
