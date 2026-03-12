# Integration Patterns — Phantom + Helius

End-to-end patterns for building frontend Solana applications combining Phantom Connect SDK with Helius infrastructure.

## Pattern 1: Swap UI

**Architecture**: Aggregator API provides serialized transaction → Phantom signs → Helius Sender submits → poll confirmation.

This pattern is aggregator-agnostic — works with Jupiter, DFlow, or any API that returns a serialized transaction.

```tsx
import { useSolana, useAccounts } from '@phantom/react-sdk';
import { useState } from 'react';

function SwapButton({ serializedTransaction }: { serializedTransaction: string }) {
  const { solana } = useSolana();
  const { isConnected, addresses } = useAccounts();
  const [status, setStatus] = useState<'idle' | 'signing' | 'submitting' | 'confirming' | 'done' | 'error'>('idle');

  async function handleSwap() {
    if (!isConnected || !solana) return;

    try {
      // 1. Decode the transaction from the swap API
      setStatus('signing');
      const txBytes = Uint8Array.from(atob(serializedTransaction), (c) => c.charCodeAt(0));

      // 2. Phantom signs (accepts raw transaction bytes, does NOT send)
      const signedTx = await solana.signTransaction(txBytes);

      // 3. Submit to Helius Sender (browser-safe, no API key)
      setStatus('submitting');
      const base64Tx = btoa(String.fromCharCode(...new Uint8Array(signedTx)));

      const response = await fetch('https://sender.helius-rpc.com/fast', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jsonrpc: '2.0',
          id: '1',
          method: 'sendTransaction',
          params: [base64Tx, { encoding: 'base64', skipPreflight: true, maxRetries: 0 }],
        }),
      });

      const result = await response.json();
      if (result.error) throw new Error(result.error.message);

      // 4. Poll for confirmation
      setStatus('confirming');
      const signature = result.result;
      await pollConfirmation(signature);

      setStatus('done');
    } catch (error: any) {
      if (error.message?.includes('User rejected')) {
        setStatus('idle'); // User cancelled
      } else {
        setStatus('error');
      }
    }
  }

  return (
    <button onClick={handleSwap} disabled={status !== 'idle'}>
      {status === 'idle' && 'Swap'}
      {status === 'signing' && 'Approve in wallet...'}
      {status === 'submitting' && 'Submitting...'}
      {status === 'confirming' && 'Confirming...'}
      {status === 'done' && 'Done!'}
      {status === 'error' && 'Error — Retry'}
    </button>
  );
}

async function pollConfirmation(signature: string): Promise<void> {
  for (let i = 0; i < 30; i++) {
    const response = await fetch('/api/rpc', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        id: '1',
        method: 'getSignatureStatuses',
        params: [[signature]],
      }),
    });
    const { result } = await response.json();
    const status = result?.value?.[0];
    if (status?.confirmationStatus === 'confirmed' || status?.confirmationStatus === 'finalized') {
      if (status.err) throw new Error('Transaction failed');
      return;
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error('Confirmation timeout');
}
```

**Key points**:
- The swap API (Jupiter, DFlow, etc.) returns a serialized transaction — you don't build it yourself
- Phantom signs the pre-built transaction via `solana.signTransaction`
- Submit via Helius Sender HTTPS endpoint (browser-safe, no API key)
- Poll confirmation through your backend proxy (needs API key)

## Pattern 2: Portfolio Viewer

**Architecture**: Phantom provides wallet address → backend proxy calls Helius DAS and Wallet API → display balances with USD values.

```tsx
import { useAccounts, useModal } from '@phantom/react-sdk';
import { useState, useEffect } from 'react';

interface TokenBalance {
  mint: string;
  name: string;
  symbol: string;
  amount: number;
  usdValue: number | null;
  imageUrl: string | null;
}

function PortfolioViewer() {
  const { isConnected, addresses } = useAccounts();
  const { open } = useModal();
  const [tokens, setTokens] = useState<TokenBalance[]>([]);
  const [solBalance, setSolBalance] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);

  const walletAddress = addresses?.find(a => a.addressType === 'solana')?.address;

  useEffect(() => {
    if (!walletAddress) {
      setTokens([]);
      setSolBalance(null);
      return;
    }

    setLoading(true);

    // Fetch portfolio via backend proxy (Helius Wallet API — 100 credits)
    fetch(`/api/helius/v1/wallet/${walletAddress}/balances?showNative=true`)
      .then((r) => r.json())
      .then((data) => {
        // Native SOL
        if (data.nativeBalance) {
          setSolBalance(data.nativeBalance.lamports / 1e9);
        }

        // Tokens sorted by USD value (descending)
        const tokenList: TokenBalance[] = (data.tokens || []).map((t: any) => ({
          mint: t.mint,
          name: t.name || 'Unknown',
          symbol: t.symbol || t.mint.slice(0, 6),
          amount: t.amount,
          usdValue: t.usdValue,
          imageUrl: t.imageUrl,
        }));

        setTokens(tokenList);
      })
      .finally(() => setLoading(false));
  }, [walletAddress]);

  if (!isConnected) {
    return (
      <div>
        <p>Connect your wallet to view your portfolio</p>
        <button onClick={open}>Connect Wallet</button>
      </div>
    );
  }

  if (loading) return <p>Loading portfolio...</p>;

  return (
    <div>
      <h2>Portfolio</h2>
      {solBalance !== null && (
        <div>SOL: {solBalance.toFixed(4)}</div>
      )}
      {tokens.map((token) => (
        <div key={token.mint}>
          <span>{token.symbol}</span>
          <span>{token.amount.toFixed(4)}</span>
          {token.usdValue && <span>${token.usdValue.toFixed(2)}</span>}
        </div>
      ))}
    </div>
  );
}
```

**Key points**:
- Phantom provides the wallet address via `useAccounts` — no signing needed for read-only operations
- All Helius API calls go through the backend proxy (`/api/helius/...`) to keep the API key server-side
- `getWalletBalances` returns tokens sorted by USD value — ideal for portfolio display
- For detailed token metadata or NFTs, supplement with DAS `getAssetsByOwner` via the proxy

## Pattern 3: Real-Time Dashboard

**Architecture**: Phantom connection → server-side Helius WebSocket → relay to client via SSE.

```tsx
// Client component
import { useAccounts } from '@phantom/react-sdk';
import { useState, useEffect } from 'react';

interface AccountUpdate {
  lamports: number;
  slot: number;
}

function RealTimeDashboard() {
  const { addresses } = useAccounts();
  const [updates, setUpdates] = useState<AccountUpdate[]>([]);
  const [balance, setBalance] = useState<number | null>(null);

  const walletAddress = addresses?.find(a => a.addressType === 'solana')?.address;

  useEffect(() => {
    if (!walletAddress) return;

    // SSE stream from our backend (which connects to Helius WS)
    const eventSource = new EventSource(
      `/api/stream?address=${walletAddress}`
    );

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      const value = data?.result?.value;
      if (value) {
        const balanceSOL = value.lamports / 1e9;
        setBalance(balanceSOL);
        setUpdates((prev) => [
          { lamports: value.lamports, slot: data.result.context?.slot },
          ...prev.slice(0, 49), // Keep last 50 updates
        ]);
      }
    };

    return () => eventSource.close();
  }, [walletAddress]);

  return (
    <div>
      <h2>Live Balance</h2>
      {balance !== null && <p>{balance.toFixed(4)} SOL</p>}
      <h3>Recent Updates</h3>
      {updates.map((u, i) => (
        <div key={i}>Slot {u.slot}: {(u.lamports / 1e9).toFixed(4)} SOL</div>
      ))}
    </div>
  );
}
```

**Server-side SSE endpoint**: See `references/frontend-security.md` for the full WebSocket relay pattern. The key idea is:
1. Server opens `wss://mainnet.helius-rpc.com/?api-key=KEY` (key stays server-side)
2. Server subscribes to `accountSubscribe` for the user's wallet address
3. Server relays notifications to the client via SSE
4. Client consumes the SSE stream with `EventSource`

## Pattern 4: Token Transfer

**Architecture**: Build `VersionedTransaction` with CU limit + CU price + transfer instruction + Jito tip → Phantom signs → Sender submits → parse confirmation.

```tsx
import { useSolana, useAccounts } from '@phantom/react-sdk';
import {
  pipe,
  createTransactionMessage,
  setTransactionMessageFeePayer,
  setTransactionMessageLifetimeUsingBlockhash,
  appendTransactionMessageInstruction,
  compileTransaction,
  address,
  lamports,
} from '@solana/kit';
import { getTransferSolInstruction } from '@solana-program/system';
import {
  getSetComputeUnitLimitInstruction,
  getSetComputeUnitPriceInstruction,
} from '@solana-program/compute-budget';

const TIP_ACCOUNTS = [
  '4ACfpUFoaSD9bfPdeu6DBt89gB6ENTeHBXCAi87NhDEE',
  'D2L6yPZ2FmmmTKPgzaMKdhu6EWZcTpLy1Vhx8uvZe7NZ',
  '9bnz4RShgq1hAnLnZbP8kbgBg1kEmcJBYQq3gQbmnSta',
  '5VY91ws6B2hMmBFRsXkoAAdsPHBJwRfBht4DXox3xkwn',
  '2nyhqdwKcJZR2vcqCyrYsaPVdAnFoJjiksCXJ7hfEYgD',
  '2q5pghRs6arqVjRvT5gfgWfWcHWmw1ZuCzphgd5KfWGJ',
  'wyvPkWjVZz1M8fHQnMMCDTQDbkManefNNhweYk5WkcF',
  '3KCKozbAaF75qEU33jtzozcJ29yJuaLJTy2jFdzUY8bT',
  '4vieeGHPYPG2MmyPRcYjdiDmmhN3ww7hsFNap8pVN3Ey',
  '4TQLFNWK8AovT1gFvda5jfw2oJeRMKEmw7aH6MGBJ3or',
];

function TransferForm() {
  const { solana } = useSolana();
  const { addresses } = useAccounts();

  async function handleTransfer(recipient: string, amountSOL: number) {
    if (!solana) return;

    const walletAddress = addresses?.find(a => a.addressType === 'solana')?.address;
    if (!walletAddress) return;

    const payer = address(walletAddress);

    // 1. Get priority fee from backend proxy
    const feeRes = await fetch('/api/rpc', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0', id: '1',
        method: 'getPriorityFeeEstimate',
        params: [{ accountKeys: [walletAddress], options: { priorityLevel: 'High' } }],
      }),
    });
    const { result: feeResult } = await feeRes.json();
    const priorityFee = Math.ceil((feeResult?.priorityFeeEstimate || 200_000) * 1.2);

    // 2. Get blockhash via proxy
    const bhRes = await fetch('/api/rpc', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0', id: '1',
        method: 'getLatestBlockhash',
        params: [{ commitment: 'confirmed' }],
      }),
    });
    const { result: bhResult } = await bhRes.json();
    const blockhash = bhResult.value;

    // 3. Build transaction
    const tipAccount = TIP_ACCOUNTS[Math.floor(Math.random() * TIP_ACCOUNTS.length)];

    const txMessage = pipe(
      createTransactionMessage({ version: 0 }),
      (m) => setTransactionMessageFeePayer(payer, m),
      (m) => setTransactionMessageLifetimeUsingBlockhash(blockhash, m),
      (m) => appendTransactionMessageInstruction(getSetComputeUnitLimitInstruction({ units: 50_000 }), m),
      (m) => appendTransactionMessageInstruction(getSetComputeUnitPriceInstruction({ microLamports: priorityFee }), m),
      (m) => appendTransactionMessageInstruction(getTransferSolInstruction({
        source: payer,
        destination: address(recipient),
        amount: lamports(BigInt(Math.floor(amountSOL * 1_000_000_000))),
      }), m),
      (m) => appendTransactionMessageInstruction(getTransferSolInstruction({
        source: payer,
        destination: address(tipAccount),
        amount: lamports(200_000n), // 0.0002 SOL Jito tip
      }), m),
    );

    const transaction = compileTransaction(txMessage);

    // 4. Sign with Phantom
    const signedTx = await solana.signTransaction(transaction);

    // 5. Submit to Sender
    const base64Tx = btoa(String.fromCharCode(...new Uint8Array(signedTx)));

    const response = await fetch('https://sender.helius-rpc.com/fast', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        id: '1',
        method: 'sendTransaction',
        params: [base64Tx, { encoding: 'base64', skipPreflight: true, maxRetries: 0 }],
      }),
    });

    const result = await response.json();
    if (result.error) throw new Error(result.error.message);

    const signature = result.result;

    // 6. Parse transaction for confirmation display
    const parseRes = await fetch('/api/helius/v0/transactions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ transactions: [signature] }),
    });
    const [parsed] = await parseRes.json();

    return { signature, description: parsed?.description };
  }

  // render form...
}
```

**Instruction ordering** (required for Sender):
1. `setComputeUnitLimit` (first)
2. `setComputeUnitPrice` (second)
3. Your instructions (middle)
4. Jito tip transfer (last)

## Pattern 5: NFT Gallery

**Architecture**: Phantom provides wallet address → backend proxy calls Helius DAS `getAssetsByOwner` → display NFT images.

```tsx
import { useAccounts, useModal } from '@phantom/react-sdk';
import { useState, useEffect } from 'react';

interface NFT {
  id: string;
  name: string;
  image: string | null;
  collection: string | null;
}

function NFTGallery() {
  const { addresses, isConnected } = useAccounts();
  const { open } = useModal();
  const [nfts, setNfts] = useState<NFT[]>([]);
  const [loading, setLoading] = useState(false);

  const walletAddress = addresses?.find(a => a.addressType === 'solana')?.address;

  useEffect(() => {
    if (!walletAddress) return;

    setLoading(true);

    // DAS getAssetsByOwner via backend proxy (10 credits/page)
    fetch('/api/rpc', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jsonrpc: '2.0',
        id: '1',
        method: 'getAssetsByOwner',
        params: {
          ownerAddress: walletAddress,
          page: 1,
          limit: 100,
          displayOptions: { showCollectionMetadata: true },
        },
      }),
    })
      .then((r) => r.json())
      .then((data) => {
        const items = data.result?.items || [];

        // Filter to NFTs only (exclude fungible tokens)
        const nftItems: NFT[] = items
          .filter((item: any) => item.interface === 'V1_NFT' || item.interface === 'ProgrammableNFT' || item.compression?.compressed)
          .map((item: any) => ({
            id: item.id,
            name: item.content?.metadata?.name || 'Unknown NFT',
            image: item.content?.links?.image || item.content?.files?.[0]?.uri || null,
            collection: item.grouping?.find((g: any) => g.group_key === 'collection')?.group_value || null,
          }));

        setNfts(nftItems);
      })
      .finally(() => setLoading(false));
  }, [walletAddress]);

  if (!isConnected) {
    return (
      <div>
        <p>Connect your wallet to view your NFTs</p>
        <button onClick={open}>Connect Wallet</button>
      </div>
    );
  }

  if (loading) return <p>Loading NFTs...</p>;

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '16px' }}>
      {nfts.map((nft) => (
        <div key={nft.id}>
          {nft.image && <img src={nft.image} alt={nft.name} style={{ width: '100%', borderRadius: '8px' }} />}
          <p>{nft.name}</p>
        </div>
      ))}
    </div>
  );
}
```

**Key points**:
- Use `content.links.image` for the NFT image URL (hosted on Arweave/IPFS, cached by DAS)
- Filter by `interface` to separate NFTs from fungible tokens
- Compressed NFTs (`compression.compressed === true`) work identically — DAS abstracts the difference
- For collection browsing, use `getAssetsByGroup` with `groupKey: "collection"` instead

## Architecture Summary

| Pattern | Phantom Role | Helius Products | API Key Needed in Browser? |
|---|---|---|---|
| Swap UI | Signs pre-built tx | Sender (submit) | No |
| Portfolio Viewer | Provides address | Wallet API, DAS (via proxy) | No — proxy |
| Real-Time Dashboard | Provides address | WebSockets (server relay) | No — server |
| Token Transfer | Signs built tx | Sender (submit), Priority Fee (via proxy) | No |
| NFT Gallery | Provides address | DAS (via proxy) | No — proxy |

In every pattern, the Helius API key stays server-side. Only the Sender HTTPS endpoint is called directly from the browser.
