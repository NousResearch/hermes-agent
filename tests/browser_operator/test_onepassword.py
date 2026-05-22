from browser_operator.onepassword import (
    choose_best_login_item,
    extract_login_metadata,
    load_env_file_values,
)


def test_choose_best_login_item_prefers_exact_url_domain_match():
    items = [
        {"id": "1", "title": "Example staging", "urls": [{"href": "https://staging.example.com"}]},
        {"id": "2", "title": "GitHub Jasur", "urls": [{"href": "https://github.com/login"}]},
        {"id": "3", "title": "Random Git", "urls": [{"href": "https://gitlab.com"}]},
    ]

    chosen = choose_best_login_item(items, "github.com")

    assert chosen is not None
    assert chosen["id"] == "2"


def test_extract_login_metadata_never_exposes_field_values():
    item = {
        "id": "abc",
        "title": "GitHub Jasur",
        "vault": {"name": "Private"},
        "urls": [{"href": "https://github.com"}],
        "fields": [
            {"id": "username", "label": "username", "purpose": "USERNAME", "value": "jasur"},
            {"id": "password", "label": "password", "purpose": "PASSWORD", "value": "super-secret"},
            {"id": "otp", "label": "one-time password", "type": "OTP", "value": "otpauth://secret"},
        ],
    }

    metadata = extract_login_metadata(item)

    assert metadata["item_id"] == "abc"
    assert metadata["title"] == "GitHub Jasur"
    assert metadata["vault"] == "Private"
    assert metadata["username_available"] is True
    assert metadata["password_available"] is True
    assert metadata["totp_available"] is True
    assert "super-secret" not in str(metadata)
    assert "otpauth" not in str(metadata)


def test_load_env_file_values_parses_simple_dotenv_without_expanding_values(tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# comment\n"
        "OP_SERVICE_ACCOUNT_TOKEN=ops_token\n"
        "QUOTED='quoted value'\n"
        "EMPTY=\n"
        "export IGNORED_EXPORT=allowed\n",
        encoding="utf-8",
    )

    values = load_env_file_values(env_path)

    assert values["OP_SERVICE_ACCOUNT_TOKEN"] == "ops_token"
    assert values["QUOTED"] == "quoted value"
    assert values["EMPTY"] == ""
    assert values["IGNORED_EXPORT"] == "allowed"
