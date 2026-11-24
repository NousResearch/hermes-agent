from gateway.session_results import _redacted
from cron.scheduler_prompt import _archive_answer


def test_redaction_recounts_valid_cron_archive_frames(monkeypatch):
    import agent.redact
    monkeypatch.setattr(agent.redact, "redact_sensitive_text",
                        lambda text: text.replace("$POSTPROXY_API_KEY", "$POSTP..._KEY"))
    prompt = "task $POSTPROXY_API_KEY"
    answer = "done $POSTPROXY_API_KEY"
    doc = (f"# Cron Job: sample\n**Prompt Characters:** {len(prompt)}\n## Prompt\n\n{prompt}\n\n"
           f"**Response Characters:** {len(answer)}\n## Response\n\n{answer}\n")
    result = _redacted({"cron_result": [True, doc, answer, None]})["cron_result"]
    assert _archive_answer(result[1]) == "done $POSTP..._KEY"
    assert "$POSTPROXY_API_KEY" not in result[1]


def test_invalid_framed_cron_archive_stays_fail_closed_after_redaction():
    doc = "**Prompt Characters:** 5\n## Prompt\n\ntask\n\n**Response Characters:** 4\n## Response\n\ndone!\n"
    saved = _redacted({"cron_result": [True, doc, "done!", None]})["cron_result"][1]
    assert _archive_answer(saved) is None
