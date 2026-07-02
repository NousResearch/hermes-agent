"""Tests for agent.i18n -- catalog parity, fallback, language resolution."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import yaml

from agent import i18n


LOCALES_DIR = Path(__file__).resolve().parents[2] / "locales"


def _load_raw(lang: str) -> dict:
    with (LOCALES_DIR / f"{lang}.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _flatten(d, prefix="") -> dict:
    flat = {}
    for k, v in (d or {}).items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten(v, key))
        else:
            flat[key] = v
    return flat


# ---------------------------------------------------------------------------
# Catalog completeness -- this is the key invariant test.  If someone adds a
# new key to en.yaml they MUST add it to every other locale, else runtime
# falls back to English for those users and defeats the feature.
# ---------------------------------------------------------------------------

def test_all_locales_exist():
    """Every supported language must have a catalog file on disk."""
    for lang in i18n.SUPPORTED_LANGUAGES:
        assert (LOCALES_DIR / f"{lang}.yaml").is_file(), f"missing locales/{lang}.yaml"


@pytest.mark.parametrize("lang", [l for l in i18n.SUPPORTED_LANGUAGES if l != "en"])
def test_catalog_keys_match_english(lang: str):
    """Every non-English catalog must have exactly the same key set as English."""
    en_keys = set(_flatten(_load_raw("en")).keys())
    lang_keys = set(_flatten(_load_raw(lang)).keys())
    missing = en_keys - lang_keys
    extra = lang_keys - en_keys
    assert not missing, f"{lang}.yaml missing keys: {sorted(missing)}"
    assert not extra, f"{lang}.yaml has keys not in en.yaml: {sorted(extra)}"


@pytest.mark.parametrize("lang", list(i18n.SUPPORTED_LANGUAGES))
def test_catalog_placeholders_match_english(lang: str):
    """Every translated value must use the same {placeholder} tokens as English.

    A mistranslated placeholder (e.g. ``{description}`` typoed as ``{descricao}``)
    would either raise KeyError at runtime or silently drop the interpolated
    value.  Pin parity at the test layer.
    """
    import re
    placeholder_re = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")
    en_flat = _flatten(_load_raw("en"))
    lang_flat = _flatten(_load_raw(lang))
    for key, en_value in en_flat.items():
        en_placeholders = set(placeholder_re.findall(en_value))
        lang_value = lang_flat.get(key, "")
        lang_placeholders = set(placeholder_re.findall(lang_value))
        assert en_placeholders == lang_placeholders, (
            f"{lang}.yaml key={key!r}: placeholders {lang_placeholders} "
            f"don't match English {en_placeholders}"
        )


# ---------------------------------------------------------------------------
# Language resolution
# ---------------------------------------------------------------------------

def test_normalize_lang_accepts_supported():
    assert i18n._normalize_lang("zh") == "zh"
    assert i18n._normalize_lang("EN") == "en"


def test_normalize_lang_accepts_aliases():
    assert i18n._normalize_lang("chinese") == "zh"
    assert i18n._normalize_lang("zh-CN") == "zh"
    assert i18n._normalize_lang("Deutsch") == "de"
    assert i18n._normalize_lang("español") == "es"
    assert i18n._normalize_lang("jp") == "ja"
    assert i18n._normalize_lang("Ukrainian") == "uk"
    assert i18n._normalize_lang("uk-UA") == "uk"
    assert i18n._normalize_lang("ua") == "uk"
    assert i18n._normalize_lang("Turkish") == "tr"
    assert i18n._normalize_lang("tr-TR") == "tr"
    assert i18n._normalize_lang("türkçe") == "tr"


def test_normalize_lang_unknown_falls_back():
    assert i18n._normalize_lang("klingon") == "en"
    assert i18n._normalize_lang("") == "en"
    assert i18n._normalize_lang(None) == "en"


def test_env_var_override(monkeypatch):
    """HERMES_LANGUAGE wins over config."""
    i18n.reset_language_cache()
    monkeypatch.setenv("HERMES_LANGUAGE", "ja")
    assert i18n.get_language() == "ja"


def test_env_var_normalized(monkeypatch):
    i18n.reset_language_cache()
    monkeypatch.setenv("HERMES_LANGUAGE", "Chinese")
    assert i18n.get_language() == "zh"


def test_default_when_nothing_set(monkeypatch):
    """With no env var and no config override, falls back to English."""
    monkeypatch.delenv("HERMES_LANGUAGE", raising=False)
    # Force config lookup to return None -- patch the cached reader.
    i18n.reset_language_cache()
    monkeypatch.setattr(i18n, "_config_language_cached", lambda: None)
    assert i18n.get_language() == "en"


# ---------------------------------------------------------------------------
# t() semantics
# ---------------------------------------------------------------------------

def test_t_explicit_lang():
    assert i18n.t("approval.denied", lang="en").endswith("Denied")
    assert i18n.t("approval.denied", lang="zh").endswith("已拒绝")
    assert i18n.t("approval.denied", lang="uk").endswith("Відхилено")
    assert i18n.t("approval.denied", lang="tr").endswith("Reddedildi")


def test_t_formats_placeholders():
    msg = i18n.t("gateway.draining", lang="en", count=3)
    assert "3" in msg


def test_t_missing_key_returns_key():
    """A missing key returns its own path -- ugly but never crashes."""
    result = i18n.t("nonexistent.key.path", lang="en")
    assert result == "nonexistent.key.path"


def test_t_missing_key_in_non_english_falls_back_to_english(tmp_path, monkeypatch):
    """If a key exists in English but not in the target locale, fall back."""
    # Stand up a fake incomplete locale under a temp locales dir.
    fake_locales = tmp_path / "locales"
    fake_locales.mkdir()
    (fake_locales / "en.yaml").write_text("foo: English Foo\n", encoding="utf-8")
    (fake_locales / "zh.yaml").write_text("# intentionally empty\n", encoding="utf-8")
    monkeypatch.setattr(i18n, "_locales_dir", lambda: fake_locales)
    i18n.reset_language_cache()
    try:
        assert i18n.t("foo", lang="zh") == "English Foo"
    finally:
        # Clear the cache on teardown so subsequent tests don't see the
        # fake "foo: English Foo" catalog instead of the real locales/*.yaml.
        i18n.reset_language_cache()


def test_t_unknown_language_uses_english():
    """Unknown lang codes normalize to English, not to a key-path fallback."""
    assert i18n.t("approval.denied", lang="klingon") == i18n.t("approval.denied", lang="en")


# ---------------------------------------------------------------------------
# _locales_dir resolution ladder -- regression for #23943 / #27632 / #35374.
# Sealed installs (Nix store venv, pip wheel) have no source tree next to
# agent/, so _locales_dir must resolve via env override or the data scheme.
# ---------------------------------------------------------------------------

def test_locales_dir_env_override_used_when_dir_exists(tmp_path, monkeypatch):
    """HERMES_BUNDLED_LOCALES wins when it points at a real directory."""
    bundled = tmp_path / "bundled-locales"
    bundled.mkdir()
    monkeypatch.setenv("HERMES_BUNDLED_LOCALES", str(bundled))
    assert i18n._locales_dir() == bundled


def test_locales_dir_env_override_ignored_when_missing(tmp_path, monkeypatch):
    """A bogus HERMES_BUNDLED_LOCALES falls through to source/wheel resolution
    instead of returning a path that doesn't exist."""
    monkeypatch.setenv("HERMES_BUNDLED_LOCALES", str(tmp_path / "does-not-exist"))
    result = i18n._locales_dir()
    assert result != tmp_path / "does-not-exist"
    # In a source checkout this is the repo-root locales dir.
    assert result.name == "locales"


def test_locales_dir_falls_back_to_data_scheme(tmp_path, monkeypatch):
    """When neither the env override nor a source-adjacent locales/ exists,
    _locales_dir uses sysconfig's data scheme (the pip-wheel layout)."""
    import sysconfig

    # No env override.
    monkeypatch.delenv("HERMES_BUNDLED_LOCALES", raising=False)

    # Force the source-adjacent path to a location with no locales/ dir.
    fake_pkg = tmp_path / "site-packages" / "agent"
    fake_pkg.mkdir(parents=True)
    monkeypatch.setattr(i18n, "__file__", str(fake_pkg / "i18n.py"))

    # Stand up a fake data scheme containing locales/.
    data_root = tmp_path / "data-scheme"
    (data_root / "locales").mkdir(parents=True)
    real_get_path = sysconfig.get_path

    def fake_get_path(name, *args, **kwargs):
        if name == "data":
            return str(data_root)
        return real_get_path(name, *args, **kwargs)

    monkeypatch.setattr(i18n.sysconfig, "get_path", fake_get_path)

    assert i18n._locales_dir() == data_root / "locales"


def test_t_resolves_real_string_in_source_checkout():
    """Sanity: in the test environment (a source checkout) t() must return a
    human string, never the bare key path. Guards against catalog-load
    regressions independent of packaging."""
    assert i18n.t("gateway.reset.header_default", lang="en") != "gateway.reset.header_default"
    assert i18n.t("gateway.status.header", lang="en") != "gateway.status.header"


# ---------------------------------------------------------------------------
# Code-usage contract -- every t() key referenced in source must exist in the
# English catalog.  The parity tests above compare catalogs to each other, so
# they cannot catch a key that is missing from EVERY catalog: the t() call
# ships anyway and users see the raw dotted key path (t() falls back to the
# key by design).  That is exactly how gateway.verbose.mode_log slipped
# through and had to be backfilled into all 16 catalogs by hand.
# ---------------------------------------------------------------------------

REPO_ROOT = LOCALES_DIR.parent
_SCAN_EXCLUDE_DIRS = {
    ".git", "node_modules", "tests", "web", "website", "apps", "ui-tui",
    "docs", "assets",
}
_PLACEHOLDER_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _iter_t_calls():
    """Yield ``(relpath, lineno, key, kwarg_names, has_star_kwargs)`` for every
    statically-analyzable ``t("dotted.key", ...)`` call in the repo.

    Only calls whose first argument is a string literal are yielded; dynamic
    keys (f-strings, variables) cannot be checked statically and are skipped.
    Recognizes ``from agent.i18n import t`` (with or without an ``as`` alias),
    ``from agent import i18n`` + ``i18n.t(...)``, and ``import agent.i18n as
    <alias>`` + ``<alias>.t(...)``, including imports nested inside functions.
    """
    for path in sorted(REPO_ROOT.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT)
        if any(part in _SCAN_EXCLUDE_DIRS for part in rel.parts):
            continue
        try:
            src = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        # Cheap pre-filter before paying for a full parse.
        if "agent.i18n" not in src and "from agent import i18n" not in src:
            continue
        try:
            tree = ast.parse(src, filename=str(path))
        except SyntaxError:
            continue

        t_names = set()
        module_aliases = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "agent.i18n":
                    for alias in node.names:
                        if alias.name == "t":
                            t_names.add(alias.asname or "t")
                elif node.module == "agent":
                    for alias in node.names:
                        if alias.name == "i18n":
                            module_aliases.add(alias.asname or "i18n")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "agent.i18n" and alias.asname:
                        module_aliases.add(alias.asname)
        if not t_names and not module_aliases:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_t_call = (
                isinstance(func, ast.Name) and func.id in t_names
            ) or (
                isinstance(func, ast.Attribute)
                and func.attr == "t"
                and isinstance(func.value, ast.Name)
                and func.value.id in module_aliases
            )
            if not is_t_call:
                continue
            if (
                not node.args
                or not isinstance(node.args[0], ast.Constant)
                or not isinstance(node.args[0].value, str)
            ):
                continue
            kwarg_names = {kw.arg for kw in node.keywords}
            yield (
                str(rel),
                node.lineno,
                node.args[0].value,
                kwarg_names - {None},
                None in kwarg_names,  # **kwargs present
            )


def test_code_t_keys_exist_in_english_catalog():
    """Every string-literal t() key in source must resolve in en.yaml.

    A miss here means the message ships as the raw dotted key path in every
    language.  Fix by adding the key to locales/en.yaml AND every other
    catalog (the parity tests then hold you to the rest).
    """
    en_keys = set(_flatten(_load_raw("en")).keys())
    calls = list(_iter_t_calls())
    # If the scanner ever goes blind (import-pattern drift), fail loudly
    # instead of green-lighting everything.  Main has 300+ static calls.
    assert len(calls) > 50, (
        f"t() usage scan found only {len(calls)} calls — scanner is likely "
        "no longer recognizing the import pattern; update _iter_t_calls()."
    )
    missing = sorted(
        f"{path}:{lineno}: t({key!r})"
        for path, lineno, key, _, _ in calls
        if key not in en_keys
    )
    assert not missing, (
        "t() keys referenced in code but missing from locales/en.yaml "
        "(users would see the raw key path):\n  " + "\n  ".join(missing)
    )


def test_code_t_calls_supply_catalog_placeholders():
    """Literal t() calls must pass every {placeholder} the English value uses.

    A missing format kwarg does not raise at runtime -- t() logs a warning
    and returns the raw template, so users see literal ``{count}`` in chat.
    Calls that forward ``**kwargs`` are skipped (not statically checkable);
    extra kwargs are fine (str.format ignores them).
    """
    en_flat = _flatten(_load_raw("en"))
    problems = []
    for path, lineno, key, kwargs, has_star_kwargs in _iter_t_calls():
        if has_star_kwargs or key not in en_flat:
            continue
        needed = set(_PLACEHOLDER_RE.findall(en_flat[key]))
        supplied = kwargs - {"lang"}
        gap = needed - supplied
        if gap:
            problems.append(
                f"{path}:{lineno}: t({key!r}) does not supply {sorted(gap)}"
            )
    assert not problems, (
        "t() calls missing format kwargs their catalog entry requires "
        "(users would see literal {placeholder} text):\n  "
        + "\n  ".join(sorted(problems))
    )
