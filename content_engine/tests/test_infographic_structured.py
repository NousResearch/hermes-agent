from infographic_content import build_structured


def test_comparison_structured():
    d = {"title": "Clubcard vs Nectar vs Aldi",
         "body_text": "Aldi: lower prices. Tesco: Clubcard. Sainsburys: Nectar points."}
    s = build_structured(d, "comparison")
    assert s["type"] == "comparison"
    assert len(s["items"]) >= 2
    assert s["title"]


def test_flowchart_steps():
    d = {"title": "Debug in 5 min",
         "body_text": "Paste the error. Paste the schema. Ask what causes it. Read the fix."}
    s = build_structured(d, "flowchart")
    assert s["type"] == "flowchart"
    assert len(s["steps"]) >= 3
