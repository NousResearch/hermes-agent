---
name: billions-identity
description: Authenticate and mint a Decentralized Identifier (DID) for Hermes Agent on the Billions Network.
version: 1.0.0
author: AgungPrabowo123
license: MIT
metadata:
  hermes:
    tags: [Web3, Identity, KYA, DePIN]
    requires_toolsets: [terminal]
---

# Billions Network KYA Identity Bridge

This skill enables Hermes Agent to authenticate and mint a Decentralized Identifier (DID) as an ERC-721 token on the Billions Network. This provides the agent with an on-chain identity, achieving parity with frameworks like OpenClaw.

## When to Use
Load this skill when the agent needs to verify its identity on-chain or interact with Web3/DePIN networks requiring Know Your Agent (KYA) compliance.

## Prerequisites
Your `~/.hermes/.env` must contain your provisioned wallet key:
`WALLET_PRIVATE_KEY="0x..."`

## Procedure
1. Install the required identity bridge dependencies in this skill's folder:
   `npm install @iden3/js-iden3-auth uuid ethers shell-quote`
2. Execute the official identity generation script to link the agent to the Billions Smart Contract.
3. The terminal will output a Base64 payload URL.
4. Complete the human verification loop via the generated pairing URL in your browser.

## Verification
The integration is successful when the agent's unique DID (`did:iden3:billions:...`) is minted and verifiable on the Billions Network Block Explorer.
