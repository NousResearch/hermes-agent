# Common Solidity Vulnerabilities

Use this reference to turn automated signals into focused Forge tests. Confirm each issue against the actual threat model and contract invariants before reporting it as a finding.

## 1. Reentrancy

An external call can re-enter a function before its state changes are complete, allowing repeated withdrawals or invariant bypasses. Write a malicious receiver contract whose fallback calls the target again; use `vm.prank` and assertions on balances/state to prove that checks-effects-interactions or a reentrancy guard prevents it.

## 2. Unchecked Return Values

Low-level `call`, `send`, and token transfers can fail without reverting when their return value is ignored. In a Forge test, make the callee fail or use a mock ERC-20 returning `false`; assert that the caller reverts or handles the failure without committing state.

## 3. `tx.origin` Authorization

Authorizing with `tx.origin` lets an attacker route a privileged user's transaction through a malicious intermediary. Create an attacker contract, call it while pranked as the privileged EOA, and assert the privileged target function rejects the intermediary unless it checks `msg.sender`.

## 4. Integer Overflow and Underflow

Solidity 0.8+ checks arithmetic by default, but `unchecked` blocks, casts, and older compiler targets can still corrupt values. Fuzz boundary inputs with Forge (`bound`, `vm.assume`) and assert arithmetic either reverts or preserves the invariant at `type(uint256).max`, zero, and narrowing-cast boundaries.

## 5. Access-Control Failures

Missing, incorrectly scoped, or improperly initialized roles can expose minting, upgrades, pauses, or withdrawals. Use `vm.prank` with unauthorized addresses to call every privileged function, then assert `vm.expectRevert` and verify authorized roles still work.

## 6. Flash-Loan Attack Surface

Manipulable spot prices, collateral values, and share accounting can be exploited within one atomic borrow-and-repay transaction. Build a test attacker that borrows liquidity, manipulates the relevant pool or oracle, calls the target, repays, and asserts profit or an invariant failure; add fuzzing across loan size and price movement.

## 7. Frontrunning and Transaction-Ordering Dependence

Public pending transactions can be copied, reordered, or sandwiched when a contract relies on stale prices, predictable commitments, or first-come allocation. Simulate two actors in successive Forge calls, vary ordering with `vm.prank`, and assert that slippage limits, commit-reveal, deadlines, or nonce checks prevent a worse outcome.

## 8. `delegatecall` Misuse

Executing untrusted implementation code with the caller's storage layout can overwrite ownership, balances, or upgrade slots. Test with a malicious implementation that writes critical storage and assert the proxy rejects unauthorized upgrades, validates implementations, and preserves storage invariants.

## 9. `selfdestruct` Risks

Destruction or forced Ether transfers can invalidate assumptions about code existence and account balances. On the project’s target EVM version, deploy a destructible helper, force Ether to the target where relevant, and assert accounting and authorization remain safe; also test that any destruction path is properly restricted.

## 10. Uninitialized Storage and Upgradeable Contracts

Uninitialized proxies, implementations, or storage pointers can let an attacker claim ownership or corrupt state. Deploy the proxy without initialization, let an attacker attempt initialization with `vm.prank`, and assert the initializer is protected, implementation initialization is disabled, and storage slots retain expected values after upgrades.
