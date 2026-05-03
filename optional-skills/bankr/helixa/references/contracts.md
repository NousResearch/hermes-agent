# Helixa Contract References

## Network

- **Chain:** Base (Chain ID: 8453)
- **RPC:** `https://mainnet.base.org`
- **Explorer:** https://basescan.org
- **Standard:** ERC-8004 (Trustless Agents)

---

## HelixaV2 (Helixa Identity NFT)

- **Address:** `0x2e3B541C59D38b84E3Bc54e977200230A204Fe60`
- **Explorer:** https://basescan.org/address/0x2e3B541C59D38b84E3Bc54e977200230A204Fe60

### Key Functions

#### mint(address, string, string, bool) → uint256
Mint a Helixa identity NFT.

| Param | Type | Description |
|-------|------|-------------|
| `agentAddress` | address | Wallet address of the agent |
| `name` | string | Agent display name (must be unique) |
| `framework` | string | Framework identifier |
| `soulbound` | bool | If true, token is non-transferable |

**Returns:** `tokenId`

```bash
cast send 0x2e3B541C59D38b84E3Bc54e977200230A204Fe60 \
  "mint(address,string,string,bool)" \
  0xAGENT_ADDRESS "MyAgent" "openclaw" false \
  --rpc-url https://mainnet.base.org \
  --private-key $PRIVATE_KEY
```

#### getAgentByAddress(address) → (uint256, string, string, ...)
Look up an agent by wallet address.

```bash
cast call 0x2e3B541C59D38b84E3Bc54e977200230A204Fe60 \
  "getAgentByAddress(address)" 0xWALLET \
  --rpc-url https://mainnet.base.org
```

#### ownerOf(uint256) → address
Standard ERC-721. Returns the owner of a token.

```bash
cast call 0x2e3B541C59D38b84E3Bc54e977200230A204Fe60 \
  "ownerOf(uint256)" 1 \
  --rpc-url https://mainnet.base.org
```

#### tokenURI(uint256) → string
Returns the metadata URI for a token.

### ABI (Common Functions)

```json
[
  "function mint(address agentAddress, string name, string framework, bool soulbound) external payable returns (uint256)",
  "function getAgentByAddress(address wallet) external view returns (uint256, string, string, bool)",
  "function ownerOf(uint256 tokenId) external view returns (address)",
  "function tokenURI(uint256 tokenId) external view returns (string)",
  "function balanceOf(address owner) external view returns (uint256)",
  "function transferFrom(address from, address to, uint256 tokenId) external"
]
```

### Mint Pricing

- **Human mint (direct contract):** 0.0025 ETH (~$5)
- **Agent mint (via API):** $1 USDC via x402

---

## $CRED Token

- **Address:** `0xAB3f23c2ABcB4E12Cc8B593C218A7ba64Ed17Ba3`
- **Chain:** Base (8453)
- **Explorer:** https://basescan.org/token/0xAB3f23c2ABcB4E12Cc8B593C218A7ba64Ed17Ba3
- **Standard:** ERC-20

### ABI (Common Functions)

```json
[
  "function balanceOf(address account) external view returns (uint256)",
  "function transfer(address to, uint256 amount) external returns (bool)",
  "function approve(address spender, uint256 amount) external returns (bool)",
  "function allowance(address owner, address spender) external view returns (uint256)",
  "function totalSupply() external view returns (uint256)"
]
```

```bash
# Check $CRED balance
cast call 0xAB3f23c2ABcB4E12Cc8B593C218A7ba64Ed17Ba3 \
  "balanceOf(address)" 0xWALLET \
  --rpc-url https://mainnet.base.org
```

---

## x402 Facilitator

- **Provider:** Dexter (`x402.dexter.cash`)
- **Payment Token:** USDC on Base
- **USDC Address (Base):** `0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913`
