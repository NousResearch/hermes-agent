from pathlib import Path

from orchard.config import Settings
from orchard.integrations import (
    delete, fetch_allowlist, get, load_catalog, secret_fields, shared_config, status,
)
from orchard.provisioner import install_fetch_helper
from orchard.secrets import LocalStore


def _settings(tmp_path: Path) -> Settings:
    s = Settings()
    s.paths.root = tmp_path / "data"
    (s.paths.home_for("alice")).mkdir(parents=True, exist_ok=True)
    cat = tmp_path / "integrations.yaml"
    cat.write_text(
        "integrations:\n"
        "  - id: gitlab\n    name: GitLab\n    icon: \"🦊\"\n    fields:\n"
        "      - {env: GITLAB_URL, label: URL, secret: false, value: \"https://git.corp\"}\n"
        "      - {env: GITLAB_TOKEN, label: Token, secret: true}\n"
    )
    s.integrations.catalog_file = str(cat)
    return s


def test_catalog_loads(tmp_path):
    s = _settings(tmp_path)
    assert [i["id"] for i in load_catalog(s)] == ["gitlab"]
    assert get(s, "gitlab")["name"] == "GitLab"
    assert get(s, "nope") is None


def test_shared_config_and_secret_fields(tmp_path):
    s = _settings(tmp_path)
    # non-secret URL is org-common config (from catalog), NOT per-employee
    assert shared_config(s) == {"GITLAB_URL": "https://git.corp"}
    # only the token is an employee-entered field
    assert [f["env"] for f in secret_fields(s, "gitlab")] == ["GITLAB_TOKEN"]


def test_status_configured_depends_on_token_only(tmp_path):
    s = _settings(tmp_path)
    store = LocalStore(s)
    g = status(s, store, "alice")[0]
    assert g["configured"] is False                      # token missing
    url = next(f for f in g["fields"] if f["env"] == "GITLAB_URL")
    assert url["shared"] is True and url["value"] == "https://git.corp"
    tok = next(f for f in g["fields"] if f["env"] == "GITLAB_TOKEN")
    assert tok["set"] is False and "value" not in tok    # secret value never returned

    store.set("alice", "GITLAB_TOKEN", "glpat-xxx")
    assert status(s, store, "alice")[0]["configured"] is True
    # the URL was never stored per-employee — tokens only
    assert store.names("alice") == ["GITLAB_TOKEN"]


def test_fetch_allowlist_from_url_fields(tmp_path):
    s = _settings(tmp_path)  # catalog has GITLAB_URL=https://git.corp + GITLAB_TOKEN
    domains = fetch_allowlist(s)["domains"]
    assert "git.corp" in domains
    assert domains["git.corp"]["token_env"] == "GITLAB_TOKEN"
    assert domains["git.corp"]["auth"] == "bearer"


def test_install_fetch_helper_writes_bin_and_allowlist(tmp_path):
    s = _settings(tmp_path)
    install_fetch_helper(s, "alice")
    home = s.paths.home_for("alice")
    helper = home / "bin" / "orchard-fetch"
    assert helper.exists()
    import stat, json
    assert stat.S_IMODE(helper.stat().st_mode) & 0o111        # executable
    allow = json.loads((home / "integrations.json").read_text())
    assert "git.corp" in allow["domains"]


def test_delete_removes_only_tokens(tmp_path):
    s = _settings(tmp_path)
    store = LocalStore(s)
    store.set("alice", "GITLAB_TOKEN", "t")
    assert delete(s, store, "alice", "gitlab") == 1      # only the token
    assert store.names("alice") == []
