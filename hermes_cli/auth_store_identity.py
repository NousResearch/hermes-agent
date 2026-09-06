"""Three-way identity for auth-store paths used by destructive grant repair.

A boolean same/different check cannot represent probe failure. Destructive
consumers (forked-OAuth heal, clone-strip of a shared alias) must refuse
when identity is unknown rather than treating the error as "two copies".
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

AuthStoreIdentity = Literal["same", "different", "unknown"]

AUTH_STORE_IDENTITY_SAME: AuthStoreIdentity = "same"
AUTH_STORE_IDENTITY_DIFFERENT: AuthStoreIdentity = "different"
AUTH_STORE_IDENTITY_UNKNOWN: AuthStoreIdentity = "unknown"


def classify_auth_store_identity(left: Path, right: Path) -> AuthStoreIdentity:
    """Classify whether two paths name one auth store or two copies.

    ``same``: symlink, hardlink, bind-mount, or identical resolved path.
    ``different``: two distinct files, or one side genuinely missing.
    ``unknown``: resolve/stat/samefile failed; callers must not mutate.
    """
    try:
        if left.resolve(strict=False) == right.resolve(strict=False):
            return AUTH_STORE_IDENTITY_SAME
    except OSError:
        pass
    except Exception:
        pass

    try:
        if left.samefile(right):
            return AUTH_STORE_IDENTITY_SAME
        return AUTH_STORE_IDENTITY_DIFFERENT
    except FileNotFoundError:
        # A genuinely missing counterpart is a real difference: there is no
        # other store to consolidate into or strip from.
        return AUTH_STORE_IDENTITY_DIFFERENT
    except OSError:
        return AUTH_STORE_IDENTITY_UNKNOWN


def paths_are_same_auth_store(left: Path, right: Path) -> bool:
    """True only when identity is proven same. Unknown is not same."""
    return classify_auth_store_identity(left, right) == AUTH_STORE_IDENTITY_SAME
