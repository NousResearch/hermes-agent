from institutional_btc_vol.candidate_triage import rank_dislocation_candidates, write_candidate_ledger


def test_rank_dislocation_candidates_assigns_priority_without_executable_language():
    dislocations = [
        {"candidate": "IBIT 7D ATM vs Deribit 7D ATM", "gross_iv_diff_vol_pts": 7.74, "confidence": "screen-only", "next_action": "quote review"},
        {"candidate": "IBIT 1D ATM vs Deribit 1D ATM", "gross_iv_diff_vol_pts": 1.2, "confidence": "screen-only", "next_action": "watch"},
        {"candidate": "IBIT 30D ATM vs Deribit 30D ATM", "gross_iv_diff_vol_pts": -5.5, "confidence": "screen-only", "next_action": "quote review"},
    ]

    ranked = rank_dislocation_candidates(dislocations, run_id="btcvol-test", as_of_cst="2026-05-15 09:00:00 CDT")

    assert [row["rank"] for row in ranked] == [1, 2, 3]
    assert ranked[0]["candidate"] == "IBIT 7D ATM vs Deribit 7D ATM"
    assert ranked[0]["priority"] == "high"
    assert ranked[0]["direction"] == "IBIT rich vs Deribit"
    assert ranked[0]["evidence_status"] == "SCREEN-ONLY · NOT EXECUTABLE"
    assert ranked[0]["recommended_workflow"] == "draft two-counterparty indicative RFQ review (internal only)"
    assert "internal only" in ranked[0]["recommended_workflow"]
    assert ranked[1]["direction"] == "IBIT cheap vs Deribit"
    assert ranked[2]["priority"] == "low"
    assert "execute" not in ranked[0]["recommended_workflow"].lower()


def test_write_candidate_ledger_writes_jsonl_with_screen_only_records(tmp_path):
    path = tmp_path / "candidate_ledger.jsonl"
    rows = rank_dislocation_candidates(
        [{"candidate": "IBIT 7D ATM vs Deribit 7D ATM", "gross_iv_diff_vol_pts": 7.74, "confidence": "screen-only", "next_action": "quote review"}],
        run_id="btcvol-test",
        as_of_cst="2026-05-15 09:00:00 CDT",
    )

    written = write_candidate_ledger(path, rows)

    assert written == path
    text = path.read_text(encoding="utf-8")
    assert "SCREEN-ONLY" in text
    assert "btcvol-test" in text
    assert "IBIT rich vs Deribit" in text
