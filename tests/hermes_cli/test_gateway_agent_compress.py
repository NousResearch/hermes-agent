"""Classic CLI ``/compress`` through GatewayAgent: main's ``compress_now`` is the gateway's ``session.mutate
compress``, and the CLI prints the owner's report (no live gateway)."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from tests.hermes_cli import test_gateway_agent
from tests.hermes_cli.test_gateway_agent import SID

make_agent = test_gateway_agent.make_agent  # the shared fixture, bound here so pytest finds it

COMPACTED = [{"role": "user", "content": "summary", "row_id": 1}, {"role": "assistant", "content": "ok", "row_id": 2}]


def six_rows():
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"} for i in range(6)]


def _compress_with_mutate_reply(make_agent, reply=None):
    """Run main's ``compress_now`` against a scripted gateway whose ``session.resume`` answers the compacted
    transcript once ``session.mutate`` (answering ``reply``, or the fake's default) has run."""
    from agent.conversation_compression_manual import CompressRequest, compress_now
    agent, client = make_agent([])
    if reply is not None:
        client.mutate_replies.append(reply)
    original = client.rpc

    async def rpc(method, *, _timeout=None, **params):
        if method == "session.resume":
            client.calls.append((method, params))
            done = any(m == "session.mutate" for m, _ in client.calls)
            return {"revision": 9, "execution_generation": 7, "messages": COMPACTED if done else []}
        return await original(method, _timeout=_timeout, **params)
    client.rpc = rpc
    return agent, client, compress_now(agent, six_rows(), CompressRequest(), task_id=SID)


def test_manual_compress_is_the_gateway_compress_mutation(make_agent):
    """Main's /compress (compress_now -> agent._compress_context) commits on the gateway and adopts
    the compacted transcript; the CLI prints main's \"Compressed: N -> M\" report from it."""
    agent, client, result = _compress_with_mutate_reply(make_agent)
    mutate = dict(client.calls)["session.mutate"]
    assert mutate["operation"] == "compress" and mutate["expected_revision"] == 9 and mutate["payload"] == {}
    assert mutate["expected_generation"] == 7
    assert result.status == "compressed"
    assert result.after_messages == [{"role": "user", "content": "summary"}, {"role": "assistant", "content": "ok"}]
    assert result.summary["headline"] == "Compressed: 6 → 2 messages"
    assert agent.context_compressor.compression_count == 1


FAILED_SUMMARY = {"_last_compress_aborted": True, "_last_summary_error": "provider exploded"}


@pytest.mark.parametrize("command, flags, shown", [
    ("/compress", FAILED_SUMMARY, "⚠️ Compression aborted: 6 messages preserved"),
    ("/compress here 1", FAILED_SUMMARY, "⚠️ Compression aborted: 6 messages preserved"),
    ("/compress", {}, "🗜️ No changes from compression: 6 messages"),
])
def test_cli_compress_prints_the_gateways_noop_report_and_changes_nothing(make_agent, capsys, command, flags, shown):
    """The owner's ``noop`` answer (a failed summary, or nothing removable) committed nothing: the CLI
    prints the owner's report, keeps its history (an aborted ``here N`` tail is not doubled or lost),
    counts no compression and leaves the status-bar estimate alone."""
    from agent.manual_compression_feedback import summarize_manual_compression
    from cli import HermesCLI
    agent, client = make_agent([])
    history = six_rows()
    report = summarize_manual_compression(history, history, 90, 90, compression_state=SimpleNamespace(**flags))
    client.mutate_replies.append({"session_id": SID, "operation": "compress", "status": "noop", "summary": report})
    agent.context_compressor.last_prompt_tokens = 229
    cli = HermesCLI.__new__(HermesCLI)
    cli.conversation_history, cli.agent, cli.session_id = six_rows(), agent, SID
    cli._busy_command = lambda *_a, **_k: nullcontext()
    cli._manual_compress(command)
    out = capsys.readouterr().out
    assert shown in out and "✅" not in out
    assert ("Reason: provider exploded" in out) is bool(flags)
    assert cli.conversation_history == history
    assert agent.context_compressor.compression_count == 0
    assert agent.context_compressor.last_prompt_tokens == 229
    assert [m for m, _ in client.calls].count("session.resume") == 1  # nothing committed: nothing adopted


def test_manual_compress_reports_the_gateways_committed_summary(make_agent):
    """A committed compression with a fallback summary keeps the owner's flags and reason, which the
    CLI's compressor view cannot know."""
    fallback = {"noop": False, "aborted": False, "refused_would_grow": False, "fallback_used": True,
                "headline": "Compressed with fallback: 6 → 2 messages", "token_line": "Approx request size: ~90 → ~30 tokens",
                "note": "Summary generation failed; Hermes used limited fallback context and removed 4 message(s)."}
    agent, _, result = _compress_with_mutate_reply(
        make_agent, {"session_id": SID, "operation": "compress", "revision": 10, "summary": fallback})
    assert result.summary == fallback
    assert len(result.after_messages) == 2
    assert agent.context_compressor.compression_count == 1
