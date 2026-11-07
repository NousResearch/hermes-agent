from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "optional-skills"
    / "blockchain"
    / "evm"
    / "scripts"
    / "evm_client.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("evm_skill", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Reference digests from Foundry's `cast keccak`; 136 bytes is one full keccak-256 block.
@pytest.mark.parametrize(
    ("data", "digest"),
    [
        (b"", "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"),
        (b"a" * 136, "a6c4d403279fe3e0af03729caada8374b5ca54d8065329a3ebcaeb4b60aa386e"),
        (b"a" * 137, "d869f639c7046b4929fc92a4d988a8b22c55fbadb802c0c66ebcd484f1915f39"),
    ],
)
def test_keccak256_matches_reference(data, digest):
    mod = load_module()

    assert mod._keccak256(data).hex() == digest


def test_selector_matches_precomputed_erc20_selectors():
    mod = load_module()

    for signature, selector in mod.ERC20_SELECTORS.items():
        assert mod._selector(signature) == selector
    assert mod._selector("transfer(address,uint256)") == "0xa9059cbb"
