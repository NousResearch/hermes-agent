"""The ``kimi-coding`` curated catalog must cover the current Kimi Code roster.

``GET https://api.kimi.com/coding/v1/models`` returns six ids for a Kimi Code
(``sk-kimi-*``) key:

    kimi-for-coding, kimi-for-coding-highspeed, k3, k3-256k, k3.1-pro, k3.1-pro-highspeed

The curated table in ``models_catalog_static`` is what the picker falls back to
whenever live ``/models`` discovery fails for this provider (#145480, #127906,
#143889 — the profile requests gzip and the probe does not always decode it), so
anything absent from that table is invisible in the picker even though the
account can call it.

``k3.1-pro``, ``k3.1-pro-highspeed`` and ``k3-256k`` shipped after the table was
written and were missing (#138375), so a user holding a perfectly valid Kimi
Code key got a picker without the newest model. This test pins the roster so the
fallback cannot silently drift again.

``k3`` is the one deliberate exception: the curated table carries its ``kimi-k3``
alias (accepted by the endpoint, see #134925), so the id counts as covered under
either spelling — but listing both would show the same model twice in the
picker.
"""

from hermes_cli.models_catalog_static import _PROVIDER_MODELS

# ids the live endpoint serves for a Kimi Code key today.
KIMI_CODE_LIVE_IDS = (
    "kimi-for-coding",
    "kimi-for-coding-highspeed",
    "k3-256k",
    "k3.1-pro",
    "k3.1-pro-highspeed",
)

# remaining live id, reachable through its documented alias in the curated table.
K3_ALIASES = ("k3", "kimi-k3")

# ids that only exist on the Moonshot platform catalog; the same provider serves
# those keys, so they must survive any Kimi Code roster update.
MOONSHOT_PLATFORM_IDS = ("kimi-k2.6", "kimi-k2.7-code", "kimi-k2-thinking")


def test_curated_catalog_covers_every_live_kimi_code_id():
    curated = _PROVIDER_MODELS["kimi-coding"]
    missing = [m for m in KIMI_CODE_LIVE_IDS if m not in curated]
    assert not missing, (
        "kimi-coding curated catalog is missing live Kimi Code ids: %s (curated: %s)"
        % (missing, curated)
    )


def test_k3_is_reachable_under_exactly_one_spelling():
    curated = _PROVIDER_MODELS["kimi-coding"]
    present = [m for m in K3_ALIASES if m in curated]
    assert present, (
        "kimi-coding curated catalog carries neither 'k3' nor its 'kimi-k3' alias "
        "(curated: %s)" % (curated,)
    )
    assert len(present) == 1, (
        "kimi-coding curated catalog lists k3 under both spellings (%s) — the "
        "picker would show the same model twice" % (present,)
    )


def test_moonshot_platform_ids_are_preserved():
    curated = _PROVIDER_MODELS["kimi-coding"]
    missing = [m for m in MOONSHOT_PLATFORM_IDS if m not in curated]
    assert not missing, (
        "kimi-coding curated catalog dropped Moonshot platform ids: %s (curated: %s)"
        % (missing, curated)
    )
