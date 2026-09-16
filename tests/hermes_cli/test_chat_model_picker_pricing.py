from hermes_cli.auth_model_picker import chat_model_picker_labels


def test_chat_picker_labels_include_in_out_prices():
    labels = chat_model_picker_labels(
        ["paid-model", "free-model"],
        {
            "paid-model": {"prompt": "0.000001", "completion": "0.000002"},
            "free-model": {"prompt": "0", "completion": "0"},
        },
        current_model="paid-model",
        sale_chrome=False,
    )
    assert any("paid-model" in row and "$" in row for row in labels)
    assert any("free-model" in row for row in labels)
    assert any("current" in row.lower() or "←" in row for row in labels)


def test_chat_picker_labels_without_pricing_are_plain_ids():
    labels = chat_model_picker_labels(["only-id"], {}, current_model="", sale_chrome=False)
    assert labels == ["only-id"]


def test_chat_picker_sale_chrome_marks_discount():
    labels = chat_model_picker_labels(
        ["sale"],
        {
            "sale": {
                "prompt": "0.000001",
                "completion": "0.000002",
                "original": {"prompt": "0.000002", "completion": "0.000004"},
            }
        },
        current_model="",
        sale_chrome=True,
    )
    assert any("sale" in row and "-" in row and "%" in row for row in labels)
