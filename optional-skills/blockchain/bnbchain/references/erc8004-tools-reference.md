# ERC-8004 agent tools reference

Register and resolve AI agents on the ERC-8004 Identity Registry. Supported networks: BSC (56), BSC Testnet (97), Ethereum, Base, Polygon, and their testnets where the official registry is deployed.

**agentURI** must point to a JSON metadata file following the Agent Metadata Profile: name, description, image, and `services` (e.g. MCP endpoint).

---

## register_erc8004_agent **(write)**

Register an agent on the ERC-8004 Identity Registry. Mints an on-chain agent identity (NFT) and returns the agent ID.

| Parameter | Type | Description |
|-----------|------|-------------|
| privateKey | string | Hex private key (or set `PRIVATE_KEY` in MCP env) |
| agentURI | string | URI of agent metadata (e.g. `ipfs://...`, `https://...`, or `data:application/json,...`) |
| network | string | **Required.** e.g. `bsc`, `bsc-testnet`, `ethereum`, `base`. No default — if user does not specify, ask before calling. Do not default to mainnet. |

**Returns:** `agentId` (string), `txHash`, `network`.

**Example:** After registration, owners can verify on 8004scan (mainnet or testnet).

---

## set_erc8004_agent_uri **(write)**

Update the metadata URI for an existing ERC-8004 agent. Caller must be the owner of the agent NFT.

| Parameter | Type | Description |
|-----------|------|-------------|
| privateKey | string | Owner’s private key |
| agentId | string or number | ERC-8004 agent ID (token ID from the registry) |
| newURI | string | New metadata URI (AgentURI format) |
| network | string | **Required.** Network name or chain ID. No default for this write — ask if unspecified. |

**Returns:** `success`, `txHash`, `agentId`, `network`.

---

## get_erc8004_agent **(read-only)**

Get agent info from the ERC-8004 Identity Registry: owner address and tokenURI (metadata URI).

| Parameter | Type | Description |
|-----------|------|-------------|
| agentId | string or number | ERC-8004 agent ID (token ID) |
| network | string | Network name or chain ID (optional; default `bsc`) |

**Returns:** `agentId`, `owner`, `tokenURI`, `network`.

---

## get_erc8004_agent_wallet **(read-only)**

Get the verified payment wallet address for an ERC-8004 agent (for x402 / agent payments). Set on-chain via setAgentWallet; defaults to owner on registration.

| Parameter | Type | Description |
|-----------|------|-------------|
| agentId | string or number | ERC-8004 agent ID (token ID) |
| network | string | Network name or chain ID (optional; default `bsc`) |

**Returns:** `agentId`, `agentWallet`, `network`.
