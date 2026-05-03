# Troubleshooting

## Common Issues

### RPC Rate Limits

**Symptom**: Requests failing, slow responses, or "rate limit exceeded" errors.

**Cause**: Veil queries a lot of blockchain data (UTXOs, merkle proofs, deposit queues). Public RPCs have strict rate limits.

**Solution**: Use a dedicated RPC from [Alchemy](https://www.alchemy.com/), [Infura](https://www.infura.io/), or similar:

```bash
mkdir -p ~/.clawdbot/skills/veil
echo "RPC_URL=https://base-mainnet.g.alchemy.com/v2/YOUR_KEY" > ~/.clawdbot/skills/veil/.env
chmod 600 ~/.clawdbot/skills/veil/.env
```

### VEIL_KEY_MISSING

**Symptom**: Error `VEIL_KEY required`

**Solution**: Run `veil init` to generate a keypair, or ensure your `.env.veil` file exists:

```bash
scripts/veil-init.sh
```

### USER_NOT_REGISTERED

**Symptom**: Transfer fails with "recipient not registered"

**Cause**: The recipient address hasn't registered their deposit key with Veil.

**Solution**: The recipient must run `veil register` before they can receive private transfers.

### NO_UTXOS

**Symptom**: Withdraw/transfer fails with "no UTXOs available"

**Cause**: Your deposits are still in the queue (pending) and haven't been processed into the privacy pool yet.

**Solution**: Wait for the Veil deposit engine to process your deposits. Check status with:

```bash
scripts/veil-balance.sh --address 0xYOUR_ADDRESS
scripts/veil-balance.sh --address 0xYOUR_ADDRESS --pool usdc
```

Look at the `queue` vs `private` balances in the output.

### INSUFFICIENT_BALANCE

**Symptom**: Transaction fails due to insufficient balance

**Solution**: Check your balances and ensure you have enough in the private pool (not just the queue):

```bash
scripts/veil-balance.sh --address 0xYOUR_ADDRESS
scripts/veil-balance.sh --address 0xYOUR_ADDRESS --pool usdc
```

### ERC20 Approval Failures (USDC)

**Symptom**: USDC deposit fails at the approval step.

**Cause**: The wallet may not have enough USDC balance, or there may be an existing allowance conflict.

**Solution**:
1. Verify you have sufficient token balance in your Bankr wallet.
2. If depositing via Bankr, the `veil-deposit-via-bankr.sh` script automatically sends the approval transaction first. Ensure the first (approve) transaction completes before the second (deposit) is submitted.
3. If using `--unsigned` directly, note that USDC returns a JSON array of two transactions — submit the `step: "approve"` tx first, wait for confirmation, then submit the `step: "deposit"` tx.

### Wrong Pool / Asset Mismatch

**Symptom**: Balance shows zero but you know you deposited.

**Cause**: You may be checking the wrong pool. ETH deposits go to the ETH pool, USDC to the USDC pool, etc.

**Solution**: Specify the correct pool:

```bash
scripts/veil-balance.sh --address 0xYOUR_ADDRESS --pool usdc
```

### Bankr API Errors

**Symptom**: `apiKey missing` or authentication errors

**Solution**: Ensure Bankr is configured:

```bash
cat ~/.clawdbot/skills/bankr/config.json
# Should contain: {"apiKey": "bk_...", "apiUrl": "https://api.bankr.bot"}
```

### Scripts Not Executable

**Symptom**: `Permission denied` when running scripts

**Solution**:

```bash
chmod +x scripts/*.sh
```

## Debugging Tips

1. **Check balances first** — Most issues stem from funds being in queue vs private pool
2. **Check the right pool** — Use `--pool usdc` to check USDC balances
3. **Use `--quiet` flag** — Suppresses progress output for cleaner JSON parsing
4. **Check Bankr job status** — If a deposit via Bankr hangs, the job ID is printed for manual status checks
5. **Verify RPC connectivity** — `curl -s YOUR_RPC_URL -X POST -H "Content-Type: application/json" -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}'`
6. **ERC20 deposits need two txs** — For USDC, ensure the approval tx confirms before the deposit tx is submitted

## Getting Help

- Veil SDK: https://github.com/veildotcash/veildotcash-sdk
- Bankr: https://bankr.bot
