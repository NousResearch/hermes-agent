"""Tests for credential exclusion during profile export.

Profile exports should NEVER include auth.json, .env, OAuth stores, credential
directory trees, private keys, or the timestamped config/env *backups* Hermes writes during normal operation
(``config.yaml.bak.<ts>``, ``config.yaml.bak-pre-migrate-xai-<ts>``,
``.env.bak-<...>``).  Users share exported profiles; leaking credentials or a
credential backup in the archive is a security issue.

Both the default-profile (``~/.hermes``) and named-profile export paths route
through :func:`_is_sensitive_export_name`, so these tests cover both.
"""

import tarfile

import pytest

from hermes_cli.profiles import (
    export_profile,
    _DEFAULT_EXPORT_EXCLUDE_ROOT,
    _is_sensitive_export_name,
)


class TestIsSensitiveExportName:
    """Unit coverage for the shared sensitive-name classifier."""

    @pytest.mark.parametrize(
        "name",
        [
            # Exact credential basenames
            ".env",
            ".envrc",
            ".claude.json",
            ".netrc",
            ".npmrc",
            ".pgpass",
            ".pypirc",
            "auth.json",
            "auth.lock",
            ".anthropic_oauth.json",
            "google_token.json",
            "google_oauth_pending.json",
            "google_oauth.json",
            "webhook_subscriptions.json",
            "bws_cache.json",
            "bws_cache.enc.json",
            "oauth_creds.json",
            ".git-credentials",
            # dotenv variants (non-template)
            ".env.local",
            ".env.production",
            ".env.bak-kiro-20260529115545",
            ".env.bak-gemini-embedding-20260506_004415",
            ".env.bak.example",
            # config backups (real-world shapes seen on disk)
            "config.yaml.bak.20260526_130938",
            "config.yaml.bak-pre-migrate-xai-20260410-040915",
            "config.yaml.bak-provider-key-cleanup-20260506_013334",
            "config.yml.bak.20260101_000000",
            "config.yaml.bak-kiro-context-20260529140131",
            # auth/config/tilde backups
            "auth.json.bak",
            "auth.json.20260101",
            "auth.json~",
            "config.yaml~",
            ".env~",
            # canonical credential-store backups
            ".anthropic_oauth.json.bak",
            "google_token.json.bak",
            "google_oauth_pending.json.backup-20260101",
            "google_oauth.json.old",
            "webhook_subscriptions.json.copy",
            "bws_cache.json.20260101",
            "bws_cache.enc.json.bak",
            "oauth_creds.json.tmp.4242.deadbeef",
            # private keys / keystores
            "id_rsa.key",
            "store.p12",
            "cert.pfx",
            "release.keystore",
            "release.jks",
            # credential-/token-looking containers
            "credentials.json",
            "credentials.json.bak",
            "client_secret.json",
            "access_token.txt",
            "refresh-tokens.yaml",
            "api_key.txt",
            "api-keys.ini",
            "secrets.yaml",
            # no-extension credential names
            "credentials",
            "id_rsa",
        ],
    )
    def test_sensitive_names_flagged(self, name):
        assert _is_sensitive_export_name(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            # Ordinary profile files
            "config.yaml",
            "config.yml",
            "SOUL.md",
            "MEMORY.md",
            "USER.md",
            "profile.yaml",
            "distribution.yaml",
            "README.md",
            "ca-bundle.pem",
            "server-cert.pem",
            # dotenv templates are safe to ship
            ".env.example",
            ".env.sample",
            ".env.template",
            ".env.dist",
            # token/secret substrings that are NOT credentials
            "tokenizer.json",
            "token_count.md",
            "secret-santa.md",
            "my-secrets-notes.md",  # .md is not a credential container
            "apikeys-guide.md",
            "backup-notes.md",
            # Credential-tree basenames are path-sensitive, not globally banned.
            "mcp-tokens",
            "pairing",
            "mcp-tokens.bak",
            "pairing.backup-20260101",
            # Backups of non-sensitive files remain portable.
            "notes.txt.bak",
            "draft.md.bak",
            "notes.txt~",
        ],
    )
    def test_safe_names_not_flagged(self, name):
        assert _is_sensitive_export_name(name) is False

    def test_case_insensitive(self):
        assert _is_sensitive_export_name("Config.YAML.BAK.20260101_000000") is True
        assert _is_sensitive_export_name("AUTH.JSON") is True
        assert _is_sensitive_export_name(".ENV.LOCAL") is True
        assert _is_sensitive_export_name("BWS_CACHE.ENC.JSON") is True
        assert _is_sensitive_export_name("MCP-TOKENS") is False


class TestCredentialExclusion:
    def test_auth_json_in_default_exclude_set(self):
        """auth.json must be in the default export exclusion set."""
        assert "auth.json" in _DEFAULT_EXPORT_EXCLUDE_ROOT

    def test_dotenv_in_default_exclude_set(self):
        """.env must be in the default export exclusion set."""
        assert ".env" in _DEFAULT_EXPORT_EXCLUDE_ROOT

    def test_named_profile_export_excludes_auth(self, tmp_path, monkeypatch):
        """Named profile export must not contain auth.json or .env."""
        profiles_root = tmp_path / "profiles"
        profile_dir = profiles_root / "testprofile"
        profile_dir.mkdir(parents=True)

        # Create a profile with credentials
        (profile_dir / "config.yaml").write_text("model: gpt-4\n")
        (profile_dir / "auth.json").write_text('{"tokens": {"access": "sk-secret"}}')
        (profile_dir / ".env").write_text("OPENROUTER_API_KEY=x\n")
        (profile_dir / "SOUL.md").write_text("I am helpful.\n")
        (profile_dir / "memories").mkdir()
        (profile_dir / "memories" / "MEMORY.md").write_text("# Memories\n")

        monkeypatch.setattr(
            "hermes_cli.profiles._get_profiles_root", lambda: profiles_root
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir", lambda n: profile_dir
        )
        monkeypatch.setattr("hermes_cli.profiles.validate_profile_name", lambda n: None)

        output = tmp_path / "export.tar.gz"
        result = export_profile("testprofile", str(output))

        # Check archive contents
        with tarfile.open(result, "r:gz") as tf:
            names = tf.getnames()

        assert any("config.yaml" in n for n in names), "config.yaml should be in export"
        assert any("SOUL.md" in n for n in names), "SOUL.md should be in export"
        assert not any("auth.json" in n for n in names), (
            "auth.json must NOT be in export"
        )
        assert not any(n.endswith("/.env") or n == ".env" for n in names), (
            ".env must NOT be in export"
        )

    def test_named_profile_export_excludes_backups_and_secrets(
        self, tmp_path, monkeypatch
    ):
        """Named profile export must drop config/env/auth backups and secrets,
        while keeping ordinary profile files (including .env.example)."""
        profiles_root = tmp_path / "profiles"
        profile_dir = profiles_root / "testprofile"
        profile_dir.mkdir(parents=True)

        # Files that MUST be excluded
        sensitive = [
            ".env",
            ".envrc",
            ".env.bak-kiro-20260529115545",
            ".env.bak.example",
            ".env.local",
            ".env~",
            "auth.json",
            "auth.lock",
            "auth.json.bak",
            "auth.json~",
            ".anthropic_oauth.json",
            ".anthropic_oauth.json.bak",
            "google_token.json",
            "google_token.json.bak",
            "google_oauth_pending.json",
            "google_oauth_pending.json.backup-20260101",
            "webhook_subscriptions.json",
            "webhook_subscriptions.json.copy",
            "bws_cache.enc.json",
            "bws_cache.enc.json.bak",
            "oauth_creds.json",
            "oauth_creds.json.tmp.4242.deadbeef",
            ".claude.json",
            ".git-credentials",
            ".netrc",
            ".netrc.bak",
            ".npmrc",
            ".pgpass",
            ".pypirc",
            "config.yaml.bak.20260526_130938",
            "config.yaml.bak-pre-migrate-xai-20260410-040915",
            "config.yaml~",
            "credentials.json",
            "credentials.json.bak",
            "client_secret.json",
        ]
        # Files that MUST survive
        kept = [
            "config.yaml",
            "SOUL.md",
            "profile.yaml",
            ".env.example",
            "README.md",
            "public-ca.pem",
            "public-ca.pem.bak",
            "public-ca.pem~",
        ]
        for name in sensitive:
            (profile_dir / name).write_text("SENSITIVE\n")
        for name in kept:
            (profile_dir / name).write_text("ok\n")
        private_key = (
            "-----BEGIN PRIVATE KEY-----\nSENSITIVE\n-----END PRIVATE KEY-----\n"
        )
        (profile_dir / "private.pem").write_text(private_key)
        (profile_dir / "private.pem.bak").write_text(private_key)
        (profile_dir / "private.pem.20260101").write_text(private_key)
        (profile_dir / "private.pem~").write_text(private_key)
        (profile_dir / "late-private.pem").write_text("x" * 9000 + private_key)
        public_cert = "-----BEGIN CERTIFICATE-----\nPUBLIC\n-----END CERTIFICATE-----\n"
        (profile_dir / "public-ca.pem").write_text(public_cert)
        (profile_dir / "public-ca.pem.bak").write_text(public_cert)
        (profile_dir / "public-ca.pem~").write_text(public_cert)

        # A nested backup deep in a subdir must also be dropped.
        nested = profile_dir / "skins" / "old"
        nested.mkdir(parents=True)
        (nested / "config.yaml.bak.20260101_000000").write_text("SENSITIVE\n")
        (nested / "theme.json").write_text("{}\n")

        # Canonical credential files can live below nested auth/cache paths.
        canonical_nested = [
            profile_dir / "auth" / "google_oauth.json",
            profile_dir / "auth" / "google_oauth.json.bak",
            profile_dir / "cache" / "bws_cache.json",
            profile_dir / "cache" / "bws_cache.json.20260101",
            profile_dir / "cache" / "bws_cache.enc.json",
            profile_dir / "cache" / "bws_cache.enc.json.bak",
        ]
        for path in canonical_nested:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("SENSITIVE\n")

        # Canonical credential trees must be excluded at their profile-root
        # locations, including legacy/new pairing paths and renamed backups.
        credential_tree_files = [
            profile_dir / "mcp-tokens" / "server.json",
            profile_dir / "mcp-tokens.bak" / "server.json",
            profile_dir / "pairing" / "device.json",
            profile_dir / "pairing.backup-20260101" / "device.json",
            profile_dir / "platforms" / "pairing" / "device.json",
            profile_dir
            / "platforms"
            / "pairing.backup-20260101"
            / "device.json",
        ]
        # Same-named directories outside canonical paths are ordinary user data
        # and must remain portable.
        ordinary_tree_files = [
            profile_dir / "plugins" / "demo" / "pairing" / "README.md",
            profile_dir
            / "plugins"
            / "demo"
            / "pairing.backup-20260101"
            / "README.md",
            profile_dir / "workspace" / "project" / "mcp-tokens" / "README.md",
            profile_dir
            / "workspace"
            / "project"
            / "mcp-tokens.bak"
            / "README.md",
        ]
        for path in credential_tree_files:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("SENSITIVE\n")
        for path in ordinary_tree_files:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("safe\n")
        (profile_dir / "plugins" / "demo" / "README.md").write_text("safe\n")

        # ``home/`` is the persistent subprocess HOME used by profile-backed
        # containers. Credential directories there must be removed without
        # dropping unrelated HOME or .config content.
        home_sensitive = [
            profile_dir / "home" / ".ssh" / "custom-production-key",
            profile_dir / "home" / ".ssh.bak" / "custom-production-key",
            profile_dir / "home" / ".aws" / "credentials",
            profile_dir / "home" / ".gnupg" / "private-keys-v1.d" / "key",
            profile_dir / "home" / ".kube" / "config",
            profile_dir / "home" / ".docker" / "config.json",
            profile_dir / "home" / ".azure" / "accessTokens.json",
            profile_dir / "home" / ".gcloud" / "credentials.db",
            profile_dir / "home" / ".config" / "gh" / "hosts.yml",
            profile_dir / "home" / ".config" / "gh.backup-20260101" / "hosts.yml",
            profile_dir / "home" / ".config" / "gcloud" / "credentials.db",
            profile_dir / "home" / ".config" / "github-copilot" / "hosts.json",
            profile_dir
            / "home"
            / ".config"
            / "github-copilot.backup-20260101"
            / "hosts.json",
            profile_dir / "home" / ".codex" / "auth.json",
            profile_dir / "home" / ".claude" / ".credentials.json",
            profile_dir / "home" / ".claude.json",
            profile_dir / "home" / ".minimax" / "credentials.json",
            profile_dir / "home" / ".qwen" / "oauth_creds.json",
            profile_dir
            / "home"
            / ".qwen"
            / "oauth_creds.json.tmp.4242.deadbeef",
            profile_dir / "home" / ".gemini" / "oauth_creds.json",
            profile_dir / "home" / "Library" / "Keychains" / "login.keychain-db",
        ]
        home_kept = [
            profile_dir / "home" / "README.md",
            profile_dir / "home" / "projects" / "demo.txt",
            profile_dir / "home" / ".config" / "editor" / "settings.json",
            profile_dir / "home" / ".config" / "editor" / "hosts.json",
            profile_dir / "home" / ".config" / "github-copilot-theme" / "settings.json",
            profile_dir / "home" / ".qwen" / "settings.json",
        ]
        for path in home_sensitive:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("SENSITIVE\n")
        for path in home_kept:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("safe\n")

        monkeypatch.setattr(
            "hermes_cli.profiles._get_profiles_root", lambda: profiles_root
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir", lambda n: profile_dir
        )
        monkeypatch.setattr("hermes_cli.profiles.validate_profile_name", lambda n: None)

        output = tmp_path / "export.tar.gz"
        result = export_profile("testprofile", str(output))

        with tarfile.open(result, "r:gz") as tf:
            archive_names = tf.getnames()
            basenames = {n.rsplit("/", 1)[-1] for n in archive_names}
            archive_name_set = set(archive_names)

        for name in sensitive:
            assert name not in basenames, f"{name} must NOT be in export"
        for name in kept:
            assert name in basenames, f"{name} should be in export"
        # Nested backup excluded, nested ordinary file kept.
        assert "theme.json" in basenames
        assert "README.md" in basenames
        assert "public-ca.pem" in basenames
        assert "public-ca.pem.bak" in basenames
        assert "public-ca.pem~" in basenames
        assert "private.pem" not in basenames
        assert "private.pem.bak" not in basenames
        assert "private.pem.20260101" not in basenames
        assert "private.pem~" not in basenames
        assert "late-private.pem" not in basenames
        assert "google_oauth.json" not in basenames
        assert "google_oauth.json.bak" not in basenames
        assert "bws_cache.json" not in basenames
        assert "bws_cache.json.20260101" not in basenames
        assert "bws_cache.enc.json" not in basenames
        assert "bws_cache.enc.json.bak" not in basenames
        for path in credential_tree_files:
            member = f"testprofile/{path.relative_to(profile_dir).as_posix()}"
            assert member not in archive_name_set, f"{member} must NOT be in export"
        for path in ordinary_tree_files:
            member = f"testprofile/{path.relative_to(profile_dir).as_posix()}"
            assert member in archive_name_set, f"{member} should be in export"
        # The only config.yaml.bak* we added are sensitive — none should survive.
        assert not any(b.startswith("config.yaml.bak") for b in basenames)
        for path in home_sensitive:
            member = f"testprofile/{path.relative_to(profile_dir).as_posix()}"
            assert member not in archive_name_set, f"{member} must NOT be in export"
        for path in home_kept:
            member = f"testprofile/{path.relative_to(profile_dir).as_posix()}"
            assert member in archive_name_set, f"{member} should be in export"

    def test_default_profile_export_excludes_backups_and_secrets(
        self, tmp_path, monkeypatch
    ):
        """Default-profile (~/.hermes) export must drop credentials and the
        config/env backups Hermes writes, while keeping ordinary files."""
        # The default profile IS the hermes home directory itself.
        default_home = tmp_path / ".hermes"
        default_home.mkdir(parents=True)

        sensitive = [
            ".env",
            ".envrc",
            ".env.bak-kiro-20260529115545",
            ".env.bak.example",
            ".env~",
            "auth.json",
            "auth.lock",
            "auth.json~",
            ".anthropic_oauth.json",
            ".anthropic_oauth.json.bak",
            "google_token.json",
            "google_token.json.bak",
            "google_oauth_pending.json",
            "google_oauth_pending.json.backup-20260101",
            "webhook_subscriptions.json",
            "webhook_subscriptions.json.copy",
            "bws_cache.enc.json",
            "bws_cache.enc.json.bak",
            "oauth_creds.json",
            "oauth_creds.json.tmp.4242.deadbeef",
            ".claude.json",
            ".git-credentials",
            ".netrc",
            ".netrc.bak",
            ".npmrc",
            ".pgpass",
            ".pypirc",
            "config.yaml.bak.20260526_130938",
            "config.yaml.bak-pre-migrate-xai-20260410-040915",
            "config.yaml~",
            "private.pem",
        ]
        kept = [
            "config.yaml",
            "SOUL.md",
            ".env.example",
            ".env.sample",
            ".env.template",
            ".env.dist",
        ]
        for name in sensitive:
            (default_home / name).write_text("SENSITIVE\n")
        for name in kept:
            (default_home / name).write_text("ok\n")
        (default_home / "memories").mkdir()
        (default_home / "memories" / "MEMORY.md").write_text("# Memories\n")

        # Root entries are constrained by the current allow-list. Put these
        # fixtures beneath allow-listed trees to prove sensitive filtering also
        # runs at deeper copytree levels.
        nested_root = default_home / "plugins" / "demo"
        nested_sensitive = [
            nested_root / ".anthropic_oauth.json",
            nested_root / ".anthropic_oauth.json.bak",
            nested_root / "google_token.json",
            nested_root / "google_token.json.bak",
            nested_root / "google_oauth_pending.json",
            nested_root / "google_oauth_pending.json.backup-20260101",
            nested_root / "auth" / "google_oauth.json",
            nested_root / "auth" / "google_oauth.json.old",
            nested_root / "cache" / "bws_cache.json",
            nested_root / "cache" / "bws_cache.json.20260101",
            nested_root / "cache" / "bws_cache.enc.json",
            nested_root / "cache" / "bws_cache.enc.json.bak",
            nested_root / "oauth_creds.json",
            nested_root / "oauth_creds.json.tmp.4242.deadbeef",
            nested_root / ".claude.json",
            nested_root / ".env~",
            nested_root / "auth.json~",
            nested_root / "config.yaml~",
            nested_root / "private.pem",
            nested_root / "private.pem.bak",
            nested_root / "private.pem.20260101",
            nested_root / "private.pem~",
            nested_root / "late-private.pem",
        ]
        for path in nested_sensitive:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("SENSITIVE\n")
        private_key = (
            "-----BEGIN ENCRYPTED PRIVATE KEY-----\nSENSITIVE\n"
            "-----END ENCRYPTED PRIVATE KEY-----\n"
        )
        (nested_root / "private.pem").write_text(private_key)
        (nested_root / "private.pem.bak").write_text(private_key)
        (nested_root / "private.pem.20260101").write_text(private_key)
        (nested_root / "private.pem~").write_text(private_key)
        (nested_root / "late-private.pem").write_text("x" * 9000 + private_key)
        public_cert = "-----BEGIN CERTIFICATE-----\nPUBLIC\n-----END CERTIFICATE-----\n"
        (nested_root / "public-ca.pem").write_text(public_cert)
        (nested_root / "public-ca.pem.bak").write_text(public_cert)
        (nested_root / "public-ca.pem~").write_text(public_cert)
        (nested_root / "README.md").write_text("safe\n")

        # A template dotenv remains exportable when it lives inside an
        # allow-listed root tree.
        template = default_home / "scripts" / "example" / ".env.example"
        template.parent.mkdir(parents=True)
        template.write_text("API_KEY=\n")

        canonical_tree_files = [
            default_home / "mcp-tokens" / "server.json",
            default_home / "mcp-tokens.bak" / "server.json",
            default_home / "pairing" / "device.json",
            default_home / "pairing.backup-20260101" / "device.json",
            default_home / "platforms" / "pairing" / "device.json",
            default_home
            / "platforms"
            / "pairing.backup-20260101"
            / "device.json",
        ]
        ordinary_tree_files = [
            nested_root / "pairing" / "README.md",
            nested_root / "pairing.backup-20260101" / "README.md",
            nested_root / "mcp-tokens" / "README.md",
            nested_root / "mcp-tokens.bak" / "README.md",
        ]
        for path in canonical_tree_files:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("SENSITIVE\n")
        for path in ordinary_tree_files:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("safe\n")

        monkeypatch.setattr(
            "hermes_cli.profiles._get_default_hermes_home", lambda: default_home
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir", lambda n: default_home
        )
        monkeypatch.setattr("hermes_cli.profiles.validate_profile_name", lambda n: None)

        output = tmp_path / "export.tar.gz"
        result = export_profile("default", str(output))

        with tarfile.open(result, "r:gz") as tf:
            names = tf.getnames()
            basenames = {n.rsplit("/", 1)[-1] for n in names}
            name_set = set(names)

        assert "config.yaml" in basenames
        assert "SOUL.md" in basenames
        assert "MEMORY.md" in basenames
        assert "default/.env.example" in names
        assert "default/.env.sample" in names
        assert "default/.env.template" in names
        assert "default/.env.dist" in names
        assert "README.md" in basenames
        assert "public-ca.pem" in basenames
        assert "public-ca.pem.bak" in basenames
        assert "public-ca.pem~" in basenames

        assert "auth.json" not in basenames
        assert "auth.lock" not in basenames
        assert "auth.json~" not in basenames
        assert ".env" not in basenames
        assert ".env.bak-kiro-20260529115545" not in basenames
        assert ".env.bak.example" not in basenames
        assert ".env~" not in basenames
        assert "config.yaml~" not in basenames
        assert "private.pem" not in basenames
        assert "private.pem.bak" not in basenames
        assert "private.pem.20260101" not in basenames
        assert "private.pem~" not in basenames
        assert "late-private.pem" not in basenames
        assert ".anthropic_oauth.json" not in basenames
        assert ".anthropic_oauth.json.bak" not in basenames
        assert "google_token.json" not in basenames
        assert "google_token.json.bak" not in basenames
        assert "google_oauth_pending.json" not in basenames
        assert "google_oauth_pending.json.backup-20260101" not in basenames
        assert "google_oauth.json" not in basenames
        assert "google_oauth.json.old" not in basenames
        assert "bws_cache.json" not in basenames
        assert "bws_cache.json.20260101" not in basenames
        assert "bws_cache.enc.json" not in basenames
        assert "bws_cache.enc.json.bak" not in basenames
        for path in canonical_tree_files:
            member = f"default/{path.relative_to(default_home).as_posix()}"
            assert member not in name_set, f"{member} must NOT be in export"
        for path in ordinary_tree_files:
            member = f"default/{path.relative_to(default_home).as_posix()}"
            assert member in name_set, f"{member} should be in export"
        assert not any(b.startswith("config.yaml.bak") for b in basenames)
