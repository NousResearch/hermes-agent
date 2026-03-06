"""
utils/validator.py
------------------
Solana wallet address validation for the nft-analytics skill.

Solana public keys are base-58 encoded, 32–44 characters long, and use the
Bitcoin base-58 alphabet (digits 1-9, uppercase A-H J-N P-Z, lowercase a-k m-z;
the characters 0, O, I, l are excluded to avoid visual ambiguity).
"""

from __future__ import annotations

import re

# Bitcoin base-58 alphabet, anchored to full string
_B58_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")


def validate_wallet(address: str) -> bool:
    """
    Return ``True`` if *address* matches the Solana base-58 public key format.

    This is a **format check only** — it does not verify the key exists
    on-chain or belongs to any particular account type.

    Parameters
    ----------
    address : str
        The wallet address to validate.

    Returns
    -------
    bool

    Examples
    --------
    >>> validate_wallet("9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM")
    True
    >>> validate_wallet("not-valid!!")
    False
    """
    if not address or not isinstance(address, str):
        return False
    return bool(_B58_RE.match(address.strip()))


def assert_valid_wallet(address: str) -> str:
    """
    Return the stripped address if it is valid; raise ``ValueError`` otherwise.

    Parameters
    ----------
    address : str

    Returns
    -------
    str
        The trimmed, validated address.

    Raises
    ------
    ValueError
        If the address format does not match the Solana base-58 spec.

    Examples
    --------
    >>> assert_valid_wallet("  9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM  ")
    '9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM'
    """
    addr = (address or "").strip()
    if not validate_wallet(addr):
        raise ValueError(
            f"Invalid Solana wallet address: '{addr}'\n"
            "Expected a 32–44 character base-58 string "
            "(alphabet: 1-9 A-H J-N P-Z a-k m-z)."
        )
    return addr
