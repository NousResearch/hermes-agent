# Spend Rails

Open before money moves. The skill body holds the card fields and the confirmation rule; this page
holds what differs by rail. No spend is `charged` until a receipt is observed.

## 1. Rails

| Rail | The quote comes from | The receipt that counts | Reversal |
| --- | --- | --- | --- |
| card checkout in a page | the order summary after shipping and tax | the confirmation page's order number, or the merchant's confirmation message | the merchant's refund or cancellation window; a card dispute is a last resort, not a plan |
| HTTP 402 or pay-per-request call | the 402 challenge: amount, currency, method, and the payee it names | a receipt header on the paid response, or the payment client's transaction id | usually none; treat it as final |
| wallet transfer | the transfer preview: the whole destination address, the network, the amount, and the network fee | the transaction id, confirmed on that network | none; a transfer to a wrong address is lost |
| booking or deposit | the price breakdown: unit price, taxes, fees, deposit against balance, and the free-cancellation deadline | the booking reference from the provider | the cancellation window, stated as a date and time |
| paid data endpoint or metered API | the price per call times the planned call count, with a ceiling on calls | the provider's usage record or invoice line for the run | usually none; stop at the ceiling |
| subscription or renewal | the first charge, the renewal amount, the cadence, and the next charge date | the provider's invoice or subscription receipt | cancellation before the next charge date |

## 2. Hermes surfaces, by presence

Each is used only when present, and none replaces the person's typed confirmation in this session.

- `browser_vault` — fills a saved card only on the exact origin it was saved for, after its own
  confirmation prompt, and refuses in a session nobody can answer. Record that prompt's answer beside
  the typed yes; stop on `payment_declined` rather than retry.
- `clarify` — ask for the confirmation with the card in the question.
- `browser_exec` / `browser_cdp` — read the quote and the receipt. A capture of the confirmation page
  with its order number is a receipt; a capture of the cart is a quote.
- Hermes command approval — when the paying command raises it, record its answer. An auto-approval
  (`/yolo`, `approvals.mode: off`, a `smart`-mode self-approval) or an unflagged command is no
  confirmation.
- `stripe-link-cli` with `--request-approval` asks the person in their Link app — extra evidence
  beside the typed yes. `mpp-agent`'s `mppx <url>` pays the 402 as it runs and asks no one, so running
  it is the spend.

## 3. Unknown outcomes

An outcome nobody saw — timeout, dropped connection, reloaded page, server error after submit, a
client that exited without a result — is `unknown`, not failed. Reconcile before any retry:

1. Mark the spend `unknown` with payee, amount, time and rail.
2. Look where the charge would show: order history, confirmation message, card or wallet activity,
   payment client status, provider usage record.
3. Found: record the receipt and stop; there is nothing to retry.
4. Shown absent: a retry is a new spend — show the card again and get a new confirmation.
5. Still unknown: report it as unknown with where it was checked, and ask the person to check with
   the payee before anything is retried.

## 4. Cap ledger

Rows the person can read: date, payee, amount, currency, rail, state, receipt reference, and the
running total against the cap. Never a card number, security code, wallet key, password, one-time
code or whole receipt body. An `unknown` row counts against the cap until reconciled.
