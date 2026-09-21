# content_engine/tests/test_educational_e2e.py
import llm_generate as lg


def test_educational_topic_generates_grounded(monkeypatch):
    topic = {"pillar":"Harness Tuning","topic":"tune model routing","educational":True,
             "context":"switched to cheap-first fallback across 13 gateways","kb_snippets":[]}
    monkeypatch.setattr(lg, "_llm_configs", lambda: [{"base":"x","model":"m","key":""}])
    monkeypatch.setattr(lg, "_call_llm",
        lambda s,u,c,timeout=90: "Moved my agent to a cheap-first fallback chain across 13 gateways. Here's why it cut cost without losing quality. #buildinpublic")
    d = lg.generate_one("sahil_twitter", topic, "twitter")
    assert d and "fallback" in d["body_text"].lower()
