# Legacy helper warning

scripts/nft.py and scripts/test_nft.py are legacy TestNet-only ARC-3 tooling. They are NOT the ARC-69 implementation and must not be used to generate this workflow's JSON or MainNet transactions. Leave their genesis/network guards intact; do not bypass them.

The active ARC-69 implementation is workspace/. Read SKILL.md, workspace/README.md and references/mainnet-minting.md. Metadata goes in the creation transaction note, the ASA URL points directly to media, and no metadata file/hash/attestation workflow is required.

Use the browser Pera flow for MainNet signatures; never import user recovery words/private keys into scripts or agent tools. Keep transaction review, separate broadcast, and verified confirmation. Test fixtures and successful builds are not live network evidence.
