from institutional_btc_vol.quote_workflow import build_quote_verification_demo_board


def test_quote_verification_demo_board_builds_review_only_lifecycle_from_candidates():
    candidates = [
        {
            "rank": 1,
            "candidate": "IBIT 5D ATM vs Deribit 4D ATM",
            "gross_iv_diff_vol_pts": 3.47,
            "priority": "high",
            "direction": "IBIT rich vs Deribit",
            "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
        },
        {
            "rank": 2,
            "candidate": "IBIT 11D ATM vs Deribit 14D ATM",
            "gross_iv_diff_vol_pts": -3.76,
            "priority": "medium",
            "direction": "IBIT cheap vs Deribit",
            "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
        },
    ]

    board = build_quote_verification_demo_board(candidates)

    assert board["title"] == "Quote Verification Demo Board"
    assert board["evidence_status"] == "SCREEN-ONLY · NOT EXECUTABLE"
    assert board["control"] == "Manual demo workflow only; no RFQ is sent and no executable quote is implied."
    assert board["summary"] == {
        "screen_only": 2,
        "reviewed": 0,
        "rfq_package_drafted": 0,
        "indicative_quote_1": 0,
        "indicative_quote_2": 0,
        "quote_verified": 0,
        "trade_verified": 0,
    }
    first = board["rows"][0]
    assert first["candidate"] == "IBIT 5D ATM vs Deribit 4D ATM"
    assert first["stage"] == "screen_only"
    assert first["next_action"] == "Internal candidate review; draft RFQ package only after approval."
    assert first["publishability"] == "internal-only"
    assert first["counterparty_quotes_required"] == 2
    assert first["trade_verified_requires"] == "execution record + post-trade evidence"
    assert "execute" not in first["next_action"].lower()


def test_quote_verification_demo_board_never_promotes_without_two_indicative_quotes():
    candidates = [
        {
            "rank": 1,
            "candidate": "IBIT 7D ATM vs Deribit 8D ATM",
            "gross_iv_diff_vol_pts": 4.2,
            "priority": "high",
            "indicative_quotes": [
                {"counterparty": "Desk A", "bid_iv": 0.42, "ask_iv": 0.45, "evidence_ref": "email://demo-a"}
            ],
            "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
        }
    ]

    board = build_quote_verification_demo_board(candidates)

    row = board["rows"][0]
    assert row["stage"] == "indicative_quote_1"
    assert row["quote_count"] == 1
    assert row["next_action"] == "Collect second independent indicative quote before any quote-verified label."
    assert row["publishability"] == "internal-only"
    assert board["summary"]["quote_verified"] == 0


def test_quote_verification_demo_board_marks_quote_verified_only_with_two_complete_quotes():
    candidates = [
        {
            "rank": 1,
            "candidate": "IBIT 12D ATM vs Deribit 15D ATM",
            "gross_iv_diff_vol_pts": -5.1,
            "priority": "high",
            "indicative_quotes": [
                {"counterparty": "Desk A", "bid_iv": 0.36, "ask_iv": 0.39, "evidence_ref": "email://demo-a"},
                {"counterparty": "Desk B", "bid_iv": 0.355, "ask_iv": 0.395, "evidence_ref": "chat://demo-b"},
            ],
            "evidence_status": "SCREEN-ONLY · NOT EXECUTABLE",
        }
    ]

    board = build_quote_verification_demo_board(candidates)

    row = board["rows"][0]
    assert row["stage"] == "quote_verified"
    assert row["quote_count"] == 2
    assert row["publishability"] == "diligence-review-only"
    assert row["next_action"] == "Prepare quote-verified diligence note; still not trade-verified."
    assert board["summary"]["quote_verified"] == 1
    assert board["summary"]["trade_verified"] == 0
