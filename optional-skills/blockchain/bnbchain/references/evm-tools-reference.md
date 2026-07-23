# EVM tools reference

**`network`**: string like `bsc`, `opbnb`, `ethereum`, `base`, or chain ID. Use **get_supported_networks** to list options.

- **Read-only tools:** `network` is optional; default is `bsc`.
- **Write tools** (`transfer_native_token`, `transfer_erc20`, `transfer_nft`, `transfer_erc1155`, `approve_token_spending`, `write_contract`): **`network` is REQUIRED.** No default. If the user does not specify the network, the agent MUST ask before calling; do not default to mainnet.

Tools that require **PRIVATE_KEY** in the MCP server env are marked with **(write)**.

---

## Blocks

| Tool | Description | Parameters |
|------|-------------|------------|
| get_latest_block | Get the latest block | `network` (optional) |
| get_block_by_number | Get a block by number | `blockNumber` (string), `network` |
| get_block_by_hash | Get a block by hash | `blockHash`, `network` |

---

## Transactions

| Tool | Description | Parameters |
|------|-------------|------------|
| get_transaction | Get transaction by hash | `txHash`, `network` |
| get_transaction_receipt | Get receipt by hash | `txHash`, `network` |
| estimate_gas | Estimate gas for a tx | `to`, `value` (optional, e.g. "0.1"), `data` (optional hex), `network` |

---

## Network

| Tool | Description | Parameters |
|------|-------------|------------|
| get_chain_info | Chain ID, block number, RPC URL | `network` |
| get_supported_networks | List supported networks | (none) |

---

## Wallet and balances

| Tool | Description | Parameters |
|------|-------------|------------|
| get_address_from_private_key | Derive EVM address from private key | `privateKey` (or env PRIVATE_KEY) |
| get_native_balance | Native token balance (BNB, ETH, etc.) | `address` (optional) or `privateKey`, `network` |
| get_erc20_balance | ERC20 balance for an address | `tokenAddress`, `address`, `network` (and optionally `privateKey` for default address) |

**(Write)** — require PRIVATE_KEY in env:

| Tool | Description | Parameters |
|------|-------------|------------|
| transfer_native_token | Send native token | `privateKey`, `toAddress`, `amount` (string e.g. "0.1"), `network` |
| transfer_erc20 | Send ERC20 tokens | `privateKey`, `tokenAddress`, `toAddress`, `amount` (string), `network` |
| approve_token_spending | Approve spender for ERC20 | `privateKey`, `tokenAddress`, `spenderAddress`, `amount` (string), `network` |
| transfer_nft | Transfer ERC721 NFT | `privateKey`, `tokenAddress`, `tokenId`, `toAddress`, `network` |
| transfer_erc1155 | Transfer ERC1155 | `privateKey`, `tokenAddress`, `tokenId`, `amount`, `toAddress`, `network` |

---

## Contracts

| Tool | Description | Parameters |
|------|-------------|------------|
| is_contract | Check if address is contract or EOA | `address`, `network` |
| read_contract | Call view/pure function | `contractAddress`, `abi` (JSON array), `functionName`, `args` (optional array), `network` |
| write_contract | **(Write)** Call state-changing function | `contractAddress`, `abi`, `functionName`, `args`, `privateKey` (or env), `network` |

**read_contract:** Pass the ABI of the single function (or full contract ABI). Example args: `[]` or `["0x...", "123"]`.

---

## Tokens (ERC20)

| Tool | Description | Parameters |
|------|-------------|------------|
| get_erc20_token_info | Name, symbol, decimals | `tokenAddress`, `network` |

---

## NFT (ERC721 / ERC1155)

| Tool | Description | Parameters |
|------|-------------|------------|
| get_nft_info | ERC721 metadata, owner | `tokenAddress`, `tokenId`, `network` |
| get_erc1155_token_metadata | ERC1155 token metadata | `tokenAddress`, `tokenId`, `network` |
| check_nft_ownership | Whether address owns NFT | (Check with get_nft_info or contract read if available) |
| get_nft_balance | NFT count for address in collection | (Use read_contract with balanceOf if needed) |
| get_erc1155_balance | Balance of ERC1155 token ID for address | (Use read_contract with balanceOf if needed) |

Transfer tools: see **Wallet and balances** above (`transfer_nft`, `transfer_erc1155`).

---

## ENS

| Tool | Description | Parameters |
|------|-------------|------------|
| resolve_ens | Resolve ENS name to address | `ensName`, `network` (typically ethereum) |

Note: ENS is not supported on BSC; use on Ethereum or other chains where ENS is deployed.
