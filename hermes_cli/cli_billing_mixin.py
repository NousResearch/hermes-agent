"""Billing and subscription handlers for the interactive CLI (mixed into ``HermesCLI``).
cli.py symbols are imported LAZILY inside methods — never at module load (import cycle)."""

from __future__ import annotations

_RULE = "─" * 41

# Poll `failed` reasons → copy (default: generic line carrying the raw reason).
_CHARGE_FAILED_COPY = {
    "authentication_required": "  🔴 Your bank requires verification (3DS). Complete it on the portal to finish this purchase.",
    "payment_method_expired": "  🔴 Your card has expired. Update it on the portal.",
    "card_declined": "  🔴 Your card was declined. Try another card on the portal."}

# Submit-time BillingError codes with a fixed copy (no payload/type inspection).
_CHARGE_ERROR_COPY = {
    "no_payment_method": "  💳 No card on file — top up and manage billing on the portal.",
    "cli_billing_disabled": "  Remote spending is off for this account — a billing admin can turn it on from the portal's Hermes Agent page.",
    "role_required": "  Adding funds needs an org admin/owner. Ask an admin, or manage on the portal.",
    "idempotency_conflict": "  🔴 That charge key was already used for a different amount. Start a fresh top-up."}
_CHARGE_ERROR_COPY["remote_spending_disabled"] = _CHARGE_ERROR_COPY["cli_billing_disabled"]

# Upgrade 2xx `status` → (line, echo recoveryUrl as "Portal:"). Missing status → ambiguous.
_UPGRADE_STATUS_COPY = {
    "requires_action": ("  🟡 This upgrade needs extra verification (3DS). Finish it on the portal.", True),
    "payment_failed": ("  🔴 Your card was declined. Update your payment method on the portal and try again.", True)}

# Upgrade 2xx terminal statuses → dim ✓ copy ({name} = target tier).
_UPGRADE_OK_COPY = {
    "already_on_tier": "You are already on {name}.",
    "upgraded": "Upgraded to {name}. Your new monthly credits land in a moment."}
# Pending-change mutations → dim ✓ copy.
_PENDING_OK_COPY = {
    "schedule": "Scheduled — your plan doesn't change today. You keep it until the end of the billing period, then it switches.",
    "cancel": "Scheduled — your plan stays active until the end of the billing period, then it cancels. Nothing changes today.",
    "resume": "Undone — you stay on your current plan."}

_ALLOW_REMOTE_SPENDING_CHOICES = [
    ("yes", "Allow Remote Spending", "open your browser to authorize"), ("no", "Not now", "cancel")]

# Static modal menus (value, label, description) — order is the rendered order.
_TOPUP_MENU_CHOICES = [
    ("buy", "Add funds", "a single charge, added to your balance today"),
    ("auto", "Auto-reload", "refill automatically when your balance runs low"),
    ("limit", "Monthly limit", "show the monthly spend cap (read-only)"),
    ("portal", "Manage on portal", "open the billing page in your browser"),
    ("cancel", "Cancel", "do nothing")]
_ADD_CARD_CHOICES = [
    ("portal", "Add a card on the portal", "opens the billing page in your browser"),
    ("recheck", "I've added it — check again", "re-check for the card and continue"),
    ("cancel", "Back", "do nothing")]
_AUTO_RELOAD_TOP_CHOICES = [
    ("edit", "Edit thresholds", "change when / how much to reload"),
    ("off", "Turn off", "disable auto-reload"),
    ("cancel", "Cancel", "do nothing")]
_AUTO_RELOAD_AGREE_CHOICES = [("agree", "Agree and turn on", "enable auto-reload"), ("cancel", "Cancel", "do nothing")]
_CHANGE_PLAN_ROW = ("change", "Change plan", "upgrade or downgrade in the terminal")
_CHANGE_MENU_TAIL = [
    ("portal", "Manage on portal", "open the billing page in your browser"), ("close", "Close", "do nothing")]
_CANCEL_SUB_CHOICES = [
    ("yes", "Cancel subscription", "schedule cancellation at period end"),
    ("cancel", "Go back", "keep your plan")]
_STEPUP_STALE_MSG = (
    "  Remote Spending still isn't active for this terminal — the authorization didn't take. "
    "Retry, or make this change on the portal.")
_NO_MANAGE_URL = "No manage URL available — is your portal configured?"
_KILLSWITCH_REASON_OVERVIEW = "A billing admin can turn it on from the portal's Hermes Agent page to add funds here."
_KILLSWITCH_REASON_BUY = "A billing admin can turn it on from the portal's Hermes Agent page before adding funds."


class CLIBillingMixin:
    """Mixin holding interactive-CLI billing and subscription handlers."""

    # ── Shared helpers ──

    def _modal_choice(self, title, detail, choices):
        """Run the choice modal and return the normalized choice value."""
        raw = self._prompt_text_input_modal(title=title, detail=detail, choices=choices)
        return self._normalize_slash_confirm_choice(raw, choices)

    def _block_header(self, icon, title, *, rule=True) -> None:
        """Blank line, ``<icon> <bold title>`` via _cprint, then (optionally) the rule via print."""
        from cli import _cprint, _b
        print()
        _cprint(f"  {icon} {_b(title)}")
        if rule:
            print(f"  {_RULE}")

    def _dim(self, msg, *, icon="", lead=False) -> None:
        """Dim ``  <icon><msg>`` line via _cprint; ``lead`` prints a blank line first."""
        from cli import _cprint, _d
        if lead:
            print()
        _cprint(f"  {icon}{_d(msg)}")

    def _ok(self, msg) -> None:
        """Dim ``✓ <msg>`` success line."""
        from cli import _cprint, _DIM, _RST
        _cprint(f"  {_DIM}✓ {msg}{_RST}")

    def _print_logged_out(self, state, load_failed, cmd) -> None:
        """Logged-out / fetch-failed block shared by /subscription and /topup."""
        if state.error:
            self._dim(f"{load_failed}: {state.error}", icon="💳 ", lead=True)
        else:
            self._dim("Not logged into Nous Portal.", icon="💳 ", lead=True)
            print(f"  Run `hermes portal` to log in, then {cmd}.")

    def _print_org_line(self, state) -> None:
        """Dim ``Org: <name> · <Role>`` line (skipped when there is no org)."""
        if state.org_name:
            role = (state.role or "").title()
            _org_line = f"Org: {state.org_name}{f' · {role}' if role else ''}"
            self._dim(_org_line)

    def _try_usage_model(self):
        """Shared dollar usage model (the only source with top-up dollars); None on any failure."""
        try:
            from agent.billing_usage import build_usage_model
            return build_usage_model()
        except Exception:
            return None

    def _usage_bar_lines(self, usage, plan_name) -> list:
        """Plan + top-up bars as lines (filled = remaining). Caller picks the print fn: ordering differs per surface."""
        lines: list = []
        pb = usage.plan_bar if usage else None
        if pb is not None and pb.total_usd > 0:
            filled = max(0, min(10, round(pb.fill_fraction * 10)))
            bar = ("█" * filled) + ("░" * (10 - filled))
            pct_s = f" · {pb.pct_used}% used" if pb.pct_used is not None else ""
            label = (plan_name or "plan").ljust(8)[:8]
            lines.append(f"  {label}[{bar}]  ${pb.remaining_usd:,.2f} left of ${pb.total_usd:,.2f}{pct_s}")
        tb = usage.topup_bar if usage else None
        if tb is not None and tb.remaining_usd > 0:
            lines.append(f"  {'top-up'.ljust(8)}[{'█' * 10}]  ${tb.remaining_usd:,.2f} · never expires")
        return lines

    def _print_total_spendable(self, usage, print_fn) -> None:
        if usage and usage.has_topup and usage.total_spendable_usd is not None:
            print_fn(f"  Total spendable: ${usage.total_spendable_usd:,.2f}")

    def _step_up_remote_spending(self, *, explain, noninteractive_msg, declined_msg, not_granted_msg) -> bool:
        """"! One-time setup" step-up (explain → confirm → device-flow). True only when granted; refusals print."""
        print()
        print("  ! One-time setup")
        self._dim(explain)
        if not self._app:
            print(noninteractive_msg)
            return False
        choice = self._modal_choice("Allow Remote Spending", "Opens your browser to authorize this terminal.", _ALLOW_REMOTE_SPENDING_CHOICES)
        if choice != "yes":
            print(declined_msg)
            return False
        print("  Opening your browser to allow Remote Spending…")
        try:
            from hermes_cli.auth import step_up_nous_billing_scope
            granted = step_up_nous_billing_scope(open_browser=True)
        except Exception as exc:
            print(f"  Couldn't allow Remote Spending: {exc}")
            return False
        if not granted:
            print(not_granted_msg)
        return bool(granted)

    def _print_portal_line(self, exc) -> None:
        """``Portal: <url>`` via _cprint when the error carries a portal deep-link."""
        from cli import _cprint
        if exc is not None and exc.portal_url:
            _cprint(f"  Portal: {exc.portal_url}")

    def _open_url_in_browser(self, url: str) -> bool:
        """The one portal opener. Refuses TTY-hijacking text browsers (w3m/lynx over SSH) via the auth guard."""
        if not url:
            return False
        try:
            from hermes_cli.auth import _can_open_graphical_browser, _is_remote_session
            if _is_remote_session() or not _can_open_graphical_browser():
                return False
        except Exception:
            pass  # guard unavailable → plain best-effort open
        try:
            import webbrowser
            return bool(webbrowser.open(url))
        except Exception:
            return False

    def _open_or_print_url(self, url) -> None:
        """Open ``url`` in the browser, or print it when no graphical browser can be used."""
        if not self._open_url_in_browser(url):
            print(f"  Open this URL: {url}")

    # ── /usage — Nous balance block ──

    def _print_nous_credits_block(self) -> bool:
        """Nous dollar balance block (two bars); True if anything printed. Shared dollar model first, then
        legacy ``nous_credits_lines``. Agent-independent (TUI slash-worker has no live agent). Fail-open."""
        from cli import _cprint, _b, _d
        usage = self._try_usage_model()
        if usage is not None and usage.available:
            from agent.billing_usage import format_renews
            plan = usage.plan_name or ("Free" if usage.status == "free" else None)
            renews_display = usage.renews_display or format_renews(usage.renews_at)
            renews = f" · renews {renews_display}" if renews_display else ""
            head = [f"  {_b(f'Plan: {plan}{renews}')}"] if plan else []
            head += self._usage_bar_lines(usage, usage.plan_name)
            tail = []
            if usage.status == "free":
                tail.append(f"  {_d('> Free · free models only. Run /subscription to reach paid models.')}")
            elif usage.status == "low":
                _amt = f"${usage.total_spendable_usd:,.2f}" if usage.total_spendable_usd is not None else "under $5"
                tail.append(f"  ! Low balance · {_amt} left. Run /topup or /subscription.")
            # All via _cprint like the Plan line: print()/_cprint() flush to different buffers under
            # patch_stdout. "Total spendable" alone does not count as printed (legacy lines still follow).
            if plan:
                print()
            for ln in head:
                _cprint(ln)
            self._print_total_spendable(usage, _cprint)
            for ln in tail:
                _cprint(ln)
            if head or tail:
                return True
        from agent.account_usage import nous_credits_lines
        lines = nous_credits_lines()
        if not lines:
            return False
        print()
        for line in lines:
            print(f"  {line}")
        return True

    def _print_usage_cta(self) -> None:
        """The `/usage` call-to-action; mirrors the TUI's ``USAGE_CTA``. Nous-account only."""
        self._dim('Run /subscription to change plan · /topup to add to your balance')

    # ── /subscription — view plan + change it (CLI surface) ──

    def _show_subscription(self):
        """`/subscription` (alias `/upgrade`). Deep-links NAS's ``/manage-subscription`` (NOT Stripe); never charges."""
        from agent.subscription_view import build_subscription_state, subscription_manage_url
        state = build_subscription_state()
        if not state.logged_in:
            self._print_logged_out(state, "Could not load subscription", "/subscription")
            return
        if state.context == "team":  # no personal plan — teams run on a shared balance
            self._block_header("⚕", "Team subscription")
            self._print_org_line(state)
            print(f"  This terminal is connected to {state.org_name or 'a team org'}. Teams run on a shared")
            print("  balance · use /topup to add funds.")
            self._dim('Personal subscriptions live on your personal account.')
            return
        self._subscription_overview(state, subscription_manage_url(state))

    def _subscription_overview(self, state, manage_url):
        """Plan read block, then the action: portal hand-off (member / non-interactive), catalog (Free), change menu."""
        from cli import _cprint, _b, _d
        from agent.billing_usage import format_renews
        usage = self._try_usage_model()
        c = state.current
        is_free = not (c and c.tier_id)
        can_change = state.can_change_plan
        plan_name = (c.tier_name or c.tier_id) if c else (usage.plan_name if usage else None)
        u_status = usage.status if usage else None
        _spend = usage.total_spendable_usd if usage else None
        renews_display = usage.renews_display if usage else None
        if not renews_display and c and c.cycle_ends_at:
            renews_display = format_renews(c.cycle_ends_at)
        # Headline flags the pending change ("→ Plus" / "→ cancels"); the banner (cancel > downgrade)
        # leads so it can't read as "nothing happened".
        _flip, _trans = "", None
        if c and c.cancel_at_period_end:
            _flip = " → cancels"
            _trans = (c.tier_name or "your plan", "cancels", format_renews(c.cancellation_effective_at) or "the end of the billing period")
        elif c and c.pending_downgrade_tier_name:
            _flip = f" → {c.pending_downgrade_tier_name}"
            _trans = (c.tier_name or "your plan", c.pending_downgrade_tier_name, format_renews(c.pending_downgrade_at) or "the end of the cycle")
        _left = f" · ${_spend:,.2f} left" if _spend is not None else ""
        if u_status == "low" and _spend is not None:
            _tail = ""
        else:
            _tail = " · view only" if not can_change else (f" · renews {renews_display}" if renews_display else "")
        status = f"Plan: {plan_name}{_flip}{_left}{_tail}" if plan_name else "Plan: Free · free models only"
        # All-_cprint (blanks included) so the block orders deterministically even when piped.
        _cprint("")
        if _trans:
            _from, _to, _when = _trans
            _cprint(f"  ⏳ {_b('Scheduled change')}")
            _cprint(f"  {_from} ──▶ {_to}  {_d('· ' + _when)}")
            self._dim(f"You keep {_from} (and its credits) until then.")
            _cprint("")
        _cprint(f"  ⚕ {_b(status)}")
        print(f"  {_RULE}")
        for _bar_ln in self._usage_bar_lines(usage, plan_name):
            print(_bar_ln)
        self._print_total_spendable(usage, print)
        if is_free:
            self._dim('> Paid models need a subscription. Start one to reach them.')
        elif u_status == "low":
            _amt = f"${_spend:,.2f}" if _spend is not None else "under $5"
            _cprint(f"  ! Low balance · {_amt} left. Top up or upgrade before a mid-run cutoff.")
        self._print_org_line(state)
        print(f"  {_RULE}")
        if not can_change:
            self._dim('Plan changes need an org admin/owner.', lead=True)
            if manage_url:
                print(f"  Manage on portal: {manage_url}")
        elif not self._app:  # non-interactive (TUI slash-worker / piped): the modal can't run
            print()
            if manage_url:
                print(f"  Manage your subscription: {manage_url}")
                print("  Open it in your browser, then re-run /subscription.")
        elif is_free:  # a NEW subscription needs a fresh card → catalog + portal deep-link only
            self._subscription_free_catalog(state, manage_url)
        else:
            self._subscription_change_menu(state, manage_url)

    def _subscription_free_catalog(self, state, manage_url):
        """Free + admin + interactive: catalog → pick → portal deep-link ``plan=<tier_id>`` (a new sub needs a card)."""
        from agent.subscription_view import format_tier_row, selectable_tiers, subscription_manage_url
        tiers = selectable_tiers(state)
        if not tiers:
            self._subscription_open_portal(state, manage_url, verb="Start a subscription")
            return
        self._block_header("⚕", "Choose a plan")
        for i, t in enumerate(tiers, 1):
            print(f"  {i}. {format_tier_row(t)}")
        self._dim('Starting a subscription opens the portal to add your card.')
        choices = [(t.tier_id, format_tier_row(t), f"start {t.name} on the portal") for t in tiers]
        choices.append(("cancel", "Cancel", "do nothing"))
        raw = self._prompt_text_input_modal(title="Start a subscription", detail="Pick a plan to open it on the portal.", choices=choices)
        # Rows are numbered → accept a bare number (the normalizer only knows confirm-dialog digits).
        _digit = (raw or "").strip()
        _by_row = tiers[int(_digit) - 1] if _digit.isdigit() and 1 <= int(_digit) <= len(tiers) else None
        choice = _by_row.tier_id if _by_row else self._normalize_slash_confirm_choice(raw, choices)
        if not choice or choice == "cancel":
            print("  🟡 Cancelled. No plan started.")
            return
        tier_url = subscription_manage_url(state, tier_id=choice) or manage_url
        if not tier_url:
            self._dim(_NO_MANAGE_URL)
            return
        picked = next((t for t in tiers if t.tier_id == choice), None)
        label = picked.name if picked else "your plan"
        if self._open_url_in_browser(tier_url):
            print(f"  Opening the portal to start {label}…")
        else:
            print(f"  Open this URL to start {label}: {tier_url}")
        print("  Finish in your browser, then re-run /subscription.")

    def _subscription_open_portal(self, state, manage_url, *, verb="Manage your subscription"):
        """Open / copy the manage-subscription URL — the portal hand-off."""
        print()
        if not manage_url:
            self._dim(_NO_MANAGE_URL)
            return
        choices = [
            ("open", verb, "open the subscription page in your browser"),
            ("copy", "Copy link", "copy the manage-subscription URL to your clipboard"),
            ("cancel", "Cancel", "do nothing")]
        choice = self._modal_choice(verb, "", choices)
        if choice == "open":
            self._open_or_print_url(manage_url)
            print()
            print("  Finish in your browser, then re-run /subscription.")
        elif choice == "copy":
            try:
                self._write_osc52_clipboard(manage_url)
                print(f"  📋 Copied: {manage_url}")
            except Exception:
                print(f"  Manage URL: {manage_url}")
        else:
            print("  🟡 Cancelled.")

    def _subscription_change_menu(self, state, manage_url):
        """The in-terminal change menu for a paid admin/owner (interactive)."""
        c = state.current
        # A scheduled change makes undo the likeliest intent → promote it first. The Close row is
        # "close" (not "cancel") so typing "cancel" can't be confused with "Cancel subscription".
        if c and (c.cancel_at_period_end or c.pending_downgrade_tier_name):
            keep_name = c.tier_name or "your plan"
            head = [("keep", f"Keep {keep_name} (undo the scheduled change)", "cancel the pending change"), _CHANGE_PLAN_ROW]
        else:
            head = [_CHANGE_PLAN_ROW, ("cancel_sub", "Cancel subscription", "schedule cancellation at period end")]
        choice = self._modal_choice("Manage your subscription", "", head + _CHANGE_MENU_TAIL)
        action = {
            "change": lambda: self._subscription_pick_tier(state),
            "keep": lambda: self._subscription_apply(state, ("resume", None)),
            "cancel_sub": lambda: self._subscription_confirm_cancel(state),
            "portal": lambda: self._subscription_open_portal(state, manage_url)}.get(choice)
        if action:
            action()
        else:
            print("  🟡 Closed. No plan change.")

    def _subscription_pick_tier(self, state):
        """Tier picker → preview → confirm. Paid tiers other than current (dropping to free = cancellation)."""
        from agent.subscription_view import format_tier_row, is_upgrade, selectable_tiers
        c = state.current
        selectable = selectable_tiers(state)
        if not selectable:
            print("  No other plans are available to switch to right now.")
            return
        choices = [
            (t.tier_id, f"{format_tier_row(t)} · {'upgrade' if is_upgrade(state, t.tier_id) else 'downgrade'}", f"switch to {t.name}")
            for t in selectable]
        choices.append(("cancel", "Back", "do nothing"))
        choice = self._modal_choice("Change plan", f"Current: {c.tier_name if c else 'Free'}. Pick a plan to preview the effect.", choices)
        if not choice or choice == "cancel":
            print("  🟡 Cancelled. No plan change.")
            return
        self._subscription_preview_and_confirm(state, choice)

    def _subscription_preview_and_confirm(self, state, tier_id, *, allow_stepup=True):
        """Preview → effect → confirm+apply. ``allow_stepup=False`` (post-grant replay) never re-prompts a step-up."""
        from cli import _cprint, _b, _d
        from agent.subscription_view import is_upgrade, subscription_change_preview_from_payload, subscription_manage_url
        from hermes_cli.nous_billing import BillingError, BillingScopeRequired, post_subscription_preview
        self._dim('Checking the change…')
        try:
            payload = post_subscription_preview(subscription_type_id=tier_id)
        except BillingScopeRequired:
            if allow_stepup:
                self._subscription_handle_scope_required(state, retry=("preview", tier_id))
            else:
                print(_STEPUP_STALE_MSG)
            return
        except BillingError as exc:
            self._subscription_render_error(state, exc)
            return
        p = subscription_change_preview_from_payload(payload)
        effect = p.effect
        target = p.target_tier_name or "the selected plan"
        print()
        if effect == "no_op":
            self._dim(f"You are already on {target} — nothing to change.")
            return
        if effect not in ("charge_now", "scheduled"):
            # blocked OR unknown effect → fail SAFE (never schedule on an unrecognized string) and
            # re-offer the portal. plan= rides along only for an UPGRADE hand-off (downgrades stay native).
            _cprint(f"  🟡 {p.reason or 'This change cannot be confirmed here — manage it on the portal.'}")
            _plan = tier_id if is_upgrade(state, tier_id) else None
            _mu = subscription_manage_url(state, tier_id=_plan)
            if _mu:
                print(f"  Manage on portal: {_mu}")
            return
        _cprint(f"  {_b('Confirm plan change')}  {_d('· charged now' if effect == 'charge_now' else '· scheduled · not today')}")
        if effect == "charge_now":
            _amt = f"${p.amount_due_now_cents / 100:.2f}" if p.amount_due_now_cents is not None else None
            _charged = f"{_amt} now (prorated)" if _amt else "the prorated amount now"
            _cprint(f"  Upgrade to {target}. You will be charged {_charged}.")
            # Best-effort: name the exact card, but only when the resolver rung matches what a
            # subscription charge actually uses (subPin / customerDefault — Stripe's precedence).
            _card_line = "The card on your subscription will be charged."
            try:
                from agent.billing_view import build_billing_state
                _bs = build_billing_state(timeout=6.0)
                _c = _bs.card if _bs.logged_in else None
                if _c is not None and _c.resolved_via in ("subPin", "customerDefault"):
                    _card_line = f"{_c.masked} — the card on your subscription — will be charged."
            except Exception:
                pass
            self._dim(_card_line)
            pay_label = f"Pay {_amt} & upgrade now" if _amt else "Upgrade now (prorated charge)"
            action = ("upgrade", tier_id)
            # The money-moving row is NOT the default — a bare Enter hits "Go back", so a stray keystroke can't charge.
            confirm_choices = [("cancel", "Go back", "do not charge"), ("yes", pay_label, "charge + upgrade now")]
        else:  # scheduled (whitelisted above)
            _when = p.effective_at[:10] if (p.effective_at and len(p.effective_at) >= 10) else "the end of the billing period"
            _cprint(f"  Change to {target} — takes effect {_when}. No charge now; you keep your current plan until then.")
            pay_label = f"Schedule change to {target}"
            action = ("schedule", tier_id)
            confirm_choices = [("yes", pay_label, "apply this change"), ("cancel", "Go back", "do not change")]
        if p.monthly_credits_delta:
            self._dim(f"Monthly credits change: {p.monthly_credits_delta}.")
        if self._modal_choice(pay_label, "", confirm_choices) != "yes":
            print("  🟡 Cancelled. No plan change.")
            return
        self._subscription_apply(state, action, allow_stepup=allow_stepup)

    def _subscription_confirm_cancel(self, state):
        """Confirm, then schedule a cancellation at period end."""
        from cli import _cprint, _b, _d
        from agent.billing_usage import format_renews
        c = state.current
        _end = (format_renews(c.cycle_ends_at) if (c and c.cycle_ends_at) else None) or "the end of the billing period"
        print()
        _cprint(f"  {_b('Confirm cancellation')}  {_d('· scheduled · not today')}")
        _cprint(f"  Cancel {(c.tier_name if c else 'your plan')} — it stays active until {_end}, then won't renew.")
        self._dim('You keep your remaining credits for this period. You can resume before it ends.')
        if self._modal_choice("Cancel subscription?", "", _CANCEL_SUB_CHOICES) != "yes":
            print("  🟡 Cancelled. Your plan is unchanged.")
            return
        self._subscription_apply(state, ("cancel", None))

    def _subscription_apply(self, state, action, idempotency_key=None, *, allow_stepup=True):
        """Run ("upgrade"|"schedule", tier_id) / ("cancel"|"resume", None); scope denial → step-up + ONE replay, same key."""
        from cli import _cprint
        from hermes_cli.nous_billing import (
            BillingError, BillingTransient, BillingRemoteSpendingRevoked, BillingScopeRequired, BillingSessionRevoked,
            delete_subscription_pending_change, post_subscription_upgrade, put_subscription_pending_change)
        kind, arg = action
        key = None
        if kind == "upgrade":
            from agent.billing_view import new_idempotency_key
            key = idempotency_key or new_idempotency_key()
        try:
            if kind == "upgrade":
                res = post_subscription_upgrade(subscription_type_id=arg, idempotency_key=key) or {}
                status = res.get("status")
                name = res.get("targetTierName") or "your new plan"
                if status in _UPGRADE_OK_COPY:
                    self._ok(_UPGRADE_OK_COPY[status].format(name=name))
                elif status in _UPGRADE_STATUS_COPY:
                    line, echo_url = _UPGRADE_STATUS_COPY[status]
                    _cprint(line)
                    if echo_url and res.get("recoveryUrl"):
                        _cprint(f"  Portal: {res.get('recoveryUrl')}")
                else:  # unknown / absent 2xx status → also ambiguous, not a flat failure
                    self._subscription_render_upgrade_ambiguous(None)
                return
            pending = {
                "schedule": (put_subscription_pending_change, {"subscription_type_id": arg}),
                "cancel": (put_subscription_pending_change, {"cancel": True}),
                "resume": (delete_subscription_pending_change, {})}.get(kind)
            if pending:
                pending[0](**pending[1])
                self._ok(_PENDING_OK_COPY[kind])
            self._dim('Re-run /subscription anytime to review it.')
        except BillingScopeRequired:  # rejects BEFORE charging → route to the step-up
            if allow_stepup:
                self._subscription_handle_scope_required(state, retry=action, idempotency_key=key)
            else:
                print(_STEPUP_STALE_MSG)
        except BillingError as exc:
            # Upgrade only: deterministic PRE-charge rejections (Transient/401/403 types, 4xx codes)
            # never reached Stripe → recovery copy. Transport / 5xx is INDETERMINATE (NAS may have
            # charged) → steer to a re-check, never a blind retry (a fresh key can't dedup).
            _pre_charge = (BillingTransient, BillingSessionRevoked, BillingRemoteSpendingRevoked)
            _ambiguous = (exc.error in ("network_error", "endpoint_unavailable")
                          or exc.status is None or exc.status >= 500)
            if kind == "upgrade" and _ambiguous and not isinstance(exc, _pre_charge):
                self._subscription_render_upgrade_ambiguous(exc)
            else:
                self._subscription_render_error(state, exc)

    def _subscription_handle_scope_required(self, state, *, retry, idempotency_key=None):
        """insufficient_scope → step-up, then replay `retry` ONCE so the user never re-runs the command."""
        granted = self._step_up_remote_spending(
            explain="To change your plan from the terminal, allow Remote Spending once. It opens your browser to authorize, then your change picks up right here.",
            noninteractive_msg="  Run `hermes portal` and allow Remote Spending, then re-run /subscription.",
            declined_msg="  No change made. Allow Remote Spending when you're ready.",
            not_granted_msg="  Couldn't allow Remote Spending — an org admin or owner has to approve it for this org.")
        if not granted:
            return
        self._ok("Remote Spending allowed.")
        # Bust the 30s token cache (it still holds the pre-grant token; _request only busts on 401).
        try:
            from hermes_cli import nous_billing as _nb
            _nb.invalidate_cached_token()
        except Exception:
            pass
        # Re-fetch fresh state, then replay the held action ONCE (allow_stepup=False).
        from agent.subscription_view import build_subscription_state
        try:
            fresh = build_subscription_state()
        except Exception:
            fresh = state
        if retry[0] == "preview":
            self._subscription_preview_and_confirm(fresh, retry[1], allow_stepup=False)
        else:
            self._subscription_apply(fresh, retry, idempotency_key=idempotency_key, allow_stepup=False)

    def _subscription_render_error(self, state, exc):
        """Render a subscription BillingError (a lighter _billing_render_charge_error)."""
        from cli import _cprint
        msg = str(exc) or "Something went wrong."
        if exc.error == "insufficient_scope":  # defensive: the flow routes scope to the step-up before here
            _cprint("  🟡 Remote Spending isn't allowed yet. Allow it, then retry.")
        elif exc.error in ("subscription_mutation_rejected", "preview_rejected"):
            _cprint(f"  🟡 {msg}")
        else:
            _cprint(f"  🔴 {msg}")
        self._print_portal_line(exc)

    def _subscription_render_upgrade_ambiguous(self, exc):
        """AMBIGUOUS outcome (NAS may have charged) → steer to a re-check, never a blind retry (key isn't persisted)."""
        from cli import _cprint
        _cprint("  🟡 Couldn't confirm the upgrade — your card may or may not have been charged.")
        self._dim('Re-run /subscription to check your plan before trying again.')
        self._print_portal_line(exc)

    # ── /topup — Remote Spending (CLI surface, all 5 screens) ──

    def _show_billing(self, command: str = "/topup"):
        """`/topup` — ZERO sub-commands (argument ignored; Overview is the only route). Non-interactive never
        prompts. Money is Decimal end-to-end; the terminal never collects card details."""
        from agent.billing_view import build_billing_state
        state = build_billing_state()
        if not state.logged_in:
            self._print_logged_out(state, "Couldn't load billing", "/topup")
            return
        self._billing_overview(state)

    def _billing_portal_hint(self, state, *, reason: str = "") -> None:
        """Print a portal deep-link line (the funnel for portal-only actions)."""
        if not state.portal_url:
            return
        if reason:
            print(f"  {reason}")
        print(f"  Manage on portal: {state.portal_url}")

    def _billing_require_admin(self, state, *, icon="💳 ", off_reason=_KILLSWITCH_REASON_BUY) -> bool:
        """Admin + org kill-switch gate; portal funnel + False when blocked. ``icon`` adds a blank line + prefix."""
        if state.can_change_plan and state.cli_billing_enabled:
            return True
        if icon:
            print()
        if not state.can_change_plan:
            self._dim('Billing actions require an org admin/owner.', icon=icon)
            self._billing_portal_hint(state)
        else:
            self._dim('Remote spending is off for this org.', icon=icon)
            self._billing_portal_hint(state, reason=off_reason)
        return False

    def _billing_overview(self, state):
        """Screen 1 — balance, bars, menu. No scope preflight (a charge 403s); a missing card does NOT gate it."""
        from cli import _cprint, _b
        from agent.billing_view import format_money
        usage = self._try_usage_model()
        print()
        _cprint(f"  💳 {_b(f'Top up · balance {format_money(state.balance_usd)}')}")
        self._print_org_line(state)
        print(f"  {_RULE}")
        for _bar_ln in self._usage_bar_lines(usage, usage.plan_name if usage else None):
            print(_bar_ln)
        ar = state.auto_reload
        if ar is not None:
            if ar.enabled:
                print(f"  Auto-reload: on — below {format_money(ar.threshold_usd)} → reload to {format_money(ar.reload_to_usd)}")
            else:
                print("  Auto-reload: off")
        if state.can_change_plan and state.cli_billing_enabled:  # card at a glance, full-menu case only
            if state.card is not None:
                print(f"  Card: {state.card.display}")
            else:
                self._dim('No saved card on file — “Add funds” walks you through adding one.')
        print(f"  {_RULE}")
        # Action gating: admin + kill-switch for charge/auto-reload; everyone gets portal.
        if not self._billing_require_admin(state, icon="", off_reason=_KILLSWITCH_REASON_OVERVIEW):
            return
        if not self._app:  # non-interactive: no modal, just the portal funnel
            self._billing_portal_hint(state)
            return
        # One-time vs automatic — the distinction stated up front in each first sentence.
        self._dim('Add funds now — a single charge, added to your balance today.')
        _amounts = [ar.reload_to_usd, ar.threshold_usd] if ar is not None and ar.enabled else [None]
        if all(a is not None and a.is_finite() for a in _amounts):
            _auto_line = f"Refill when low — charges {format_money(ar.reload_to_usd)} automatically when your balance falls below {format_money(ar.threshold_usd)}."
        else:
            _auto_line = "Refill when low — charges your card automatically when your balance falls below the amount you set."
        self._dim(_auto_line)
        print(f"  {_RULE}")
        # No "Allow Remote Spending" item — discovered at pay time. "Add funds" charges the org's
        # portal-saved card (server-held; no card ref leaves the client).
        action = {
            "buy": self._billing_buy_flow,
            "auto": self._billing_auto_reload_flow,
            "limit": self._billing_limit_screen,
            "portal": self._billing_open_portal}.get(self._modal_choice("Top up your balance", "", _TOPUP_MENU_CHOICES))
        if action:
            action(state)
        else:
            print("  Cancelled.")

    def _billing_open_portal(self, state):
        if not state.portal_url:
            print("  No portal URL available.")
            return
        self._open_or_print_url(state.portal_url)
        print("  Complete billing changes in the browser.")

    def _billing_add_card_flow(self, state):
        """No card → add it on the portal (never in-terminal), bounded re-check loop. Refreshed state, or None."""
        from cli import _cprint
        self._block_header("💳", "Add a card first", rule=False)
        _cprint("  No saved card on file.")
        self._dim('Add a card once on the portal billing page — after that you can top up right from the terminal.')
        for _ in range(8):  # bounded: portal-open plus a handful of re-checks
            choice = self._modal_choice("Add a card", "", _ADD_CARD_CHOICES)
            if choice == "portal":
                self._billing_open_portal(state)
                self._dim('Add the card on the billing page, then pick “check again” here.')
            elif choice == "recheck":
                from agent.billing_view import build_billing_state
                try:
                    fresh = build_billing_state()
                except Exception:
                    fresh = None
                if fresh is not None and fresh.logged_in:
                    state = fresh
                if state.card is not None:
                    self._ok(f"Card found: {state.card.display} — continuing.")
                    return state
                print("  Still no card on file — finish adding it on the portal, then check again.")
            else:
                break
        print("  Cancelled. No funds added.")
        return None

    def _billing_buy_flow(self, state):
        """Screen 2 (presets) → Screen 3 (confirm+charge+poll). No scope preflight: react to the server's 403s."""
        from agent.billing_view import format_money, validate_charge_amount
        if not self._billing_require_admin(state):
            return
        if not self._app:
            self._block_header("💳", "Add funds", rule=False)
            print(f"  Presets: {', '.join(format_money(p) for p in state.charge_presets)}")
            print("  Run this in the interactive CLI to complete a purchase.")
            self._billing_portal_hint(state)
            return
        if state.card is None:  # guided add-card path first, so the amount pick can't 403
            state = self._billing_add_card_flow(state)
            if state is None or state.card is None:
                return
        preset_choices = [(str(p), format_money(p), "one-time credit purchase") for p in state.charge_presets]
        preset_choices.append(("custom", "Custom amount…", "enter your own amount"))
        preset_choices.append(("cancel", "Cancel", "do nothing"))
        card = state.card
        choice = self._modal_choice("Add funds", f"Payment: {card.display}" if card else "No saved card on file", preset_choices)
        if not choice or choice == "cancel":
            print("  Cancelled. No funds added.")
            return
        from decimal import Decimal
        if choice == "custom":
            entered = self._prompt_text_input("  Amount (USD): ")
            if entered is None:  # cancelled (e.g. slash-worker can't prompt off-thread)
                print("  Cancelled. No funds added.")
                return
            v = validate_charge_amount(entered or "", min_usd=state.min_usd, max_usd=state.max_usd)
            if not v.ok:
                print(f"  🔴 {v.error}")
                return
            amount = v.amount
        else:
            try:
                amount = Decimal(choice)
            except Exception:
                print("  🔴 Invalid selection.")
                return
        self._billing_confirm_and_charge(state, amount)

    def _billing_confirm_and_charge(self, state, amount):
        """Screen 3 — confirm total + consent, charge, then poll to settlement."""
        from agent.billing_view import format_money, new_idempotency_key
        card = state.card
        self._block_header("💳", "Confirm purchase")
        print(f"  Total: {format_money(amount)}")
        if card:
            print(f"  Payment: {card.display}")
            if card.provenance is None:  # older NAS without provenance → generic line
                self._dim('Your card saved on the portal will be charged.')
        print(f"  {_RULE}")
        self._dim('By confirming, you allow Nous Research to charge your card.')
        confirm_choices = [
            ("pay", f"Pay {format_money(amount)} now", "submit the charge"),
            ("portal", "Manage on portal", "manage your card / billing in the browser"),
            ("cancel", "Go back", "do not charge")]
        if not self._app:
            print("  Run in the interactive CLI to confirm a purchase.")
            return
        choice = self._modal_choice(f"Pay {format_money(amount)}?", card.display if card else "no saved card", confirm_choices)
        if choice == "portal":
            self._billing_open_portal(state)
            return
        if choice != "pay":
            print("  Cancelled. No funds added.")
            return
        key = new_idempotency_key()  # reused on the post-step-up resume so a double-submit collapses
        self._billing_submit_and_poll(
            state, amount, key, missing_msg="  🔴 No charge id returned; please check the portal.",
            status_msg="Charge submitted — confirming settlement…",
            on_scope=lambda: self._billing_handle_scope_required(state, amount=amount, idempotency_key=key))

    def _billing_submit_and_poll(self, state, amount, key, *, missing_msg, status_msg, on_scope=None):
        """POST the charge, then poll. ``on_scope`` handles a scope denial (first submit); else it renders."""
        from cli import _cprint, _d
        from hermes_cli.nous_billing import BillingError, BillingScopeRequired, post_charge
        try:
            result = post_charge(amount_usd=amount, idempotency_key=key)
        except BillingError as exc:
            if on_scope is not None and isinstance(exc, BillingScopeRequired):
                on_scope()
            else:
                self._billing_render_charge_error(state, exc)
            return
        charge_id = result.get("chargeId")
        if not charge_id:
            print(missing_msg)
            return
        _cprint(f"  {_d(status_msg)}")
        self._billing_poll_charge(state, charge_id, amount)

    def _billing_poll_charge(self, state, charge_id, amount):
        """Poll loop: 2s interval, 5-min cap, cancellable. settled = ledger truth."""
        import time as _time
        from agent.billing_view import format_money, parse_money
        from hermes_cli.nous_billing import BillingError, BillingTransient, get_charge_status
        deadline = _time.time() + 300
        while _time.time() < deadline:
            try:
                status = get_charge_status(charge_id)
            except BillingTransient as exc:  # retry-after, NOT a failure — back off and keep polling
                _time.sleep(min(exc.retry_after or 5, 30))
                continue
            except BillingError as exc:
                print(f"  🔴 Could not check the charge: {exc}")
                return
            state_str = status.get("status")
            if state_str == "settled":
                amt = status.get("amountUsd")
                print(f"  ✓ {format_money(parse_money(amt)) if amt else format_money(amount)} added to your balance.")
                return
            if state_str == "failed":
                self._billing_render_charge_failed(state, status.get("reason"))
                return
            _time.sleep(2.0)  # pending
        print("  🟡 Still processing after 5 minutes — this is a timeout, not a failure. Check /billing or the portal shortly.")
        self._billing_portal_hint(state)

    def _billing_render_charge_failed(self, state, reason):
        """Poll `failed` reasons → the right copy + portal funnel."""
        reason = (reason or "").strip()
        print(_CHARGE_FAILED_COPY.get(reason) or f"  🔴 The charge didn't go through ({reason or 'processing_error'}).")
        self._billing_portal_hint(state)

    def _billing_render_charge_error(self, state, exc):
        """Submit-time BillingError. Order matters: revoked/session before code lookups; Transient before scope."""
        from hermes_cli.nous_billing import BillingTransient, BillingRemoteSpendingRevoked, BillingSessionRevoked
        code = exc.error
        portal_url = exc.portal_url or state.portal_url
        if isinstance(exc, BillingRemoteSpendingRevoked) or code == "remote_spending_revoked":
            # This terminal's spend was revoked; recovery is reconnect.
            who = "An admin stopped this terminal's spending." if exc.actor == "admin" else "You stopped this terminal's spending."
            print(f"  🔴 {who} Reconnect to restore — run `hermes portal` to re-authorize.")
        elif isinstance(exc, BillingSessionRevoked) or code == "session_revoked":
            print("  🔴 Your session was logged out. Run `hermes portal` to log in again.")
        elif code in _CHARGE_ERROR_COPY or exc.code == "remote_spending_disabled":
            # Fixed copy by `error`; the gate's dual error/code payload may carry it in `.code` only.
            print(_CHARGE_ERROR_COPY.get(code) or _CHARGE_ERROR_COPY["cli_billing_disabled"])
        elif code == "monthly_cap_exceeded":
            remaining = (exc.payload or {}).get("remainingUsd")
            print(f"  🔴 Monthly spend cap reached — ${remaining} headroom left." if remaining is not None else "  🔴 Monthly spend cap reached.")
        elif isinstance(exc, BillingTransient):
            wait = exc.retry_after
            mins = f" (try again in ~{max(1, round(wait / 60))} min)" if wait else ""
            print(f"  🟡 Too many charges right now{mins}. This isn't a payment failure.")
        elif code == "insufficient_scope":
            # Never leak the raw billing:manage scope (a raced post-grant replay can re-raise it).
            print("  🔴 Remote Spending needs approval — run /topup to allow it, then retry.")
        else:
            print(f"  🔴 {exc}")
        if portal_url:
            print(f"  Portal: {portal_url}")

    def _billing_handle_scope_required(self, state, *, amount=None, idempotency_key=None):
        """403 insufficient_scope → reauth, then resume ``amount`` on explicit confirm, reusing the idempotency key."""
        from agent.billing_view import build_billing_state, format_money, new_idempotency_key
        amount_str = format_money(amount) if amount is not None else "your top-up"
        granted = self._step_up_remote_spending(
            explain=f"To charge from this terminal, allow Remote Spending once. It opens your browser to authorize, then {amount_str} picks up right here.",
            noninteractive_msg="  Run `hermes portal` and allow Remote Spending, then retry.",
            declined_msg="  No charge made. Run /topup when you want to allow Remote Spending.",
            not_granted_msg="  Couldn't allow Remote Spending — an org admin or owner has to approve it. Your card was not charged.")
        if not granted:
            return
        # The token now has the scope, but the ORG kill-switch is a separate gate — re-fetch /state.
        fresh = build_billing_state()
        if not (fresh.logged_in and fresh.cli_billing_enabled):
            print("  Remote Spending is allowed for this terminal, but it's still off for this org. A billing admin can turn it on from the portal's Hermes Agent page, then run /topup again.")
            self._billing_portal_hint(fresh)
            return
        if fresh.card is None:  # half-done state: say so rather than a bare "✓ enabled"
            print("  ✓ Remote Spending allowed — but there's no card on file yet.")
            self._dim('Top up and manage billing on the portal to continue.')
            self._billing_portal_hint(fresh)
            return
        if amount is None:  # scope-required hit outside a charge (e.g. auto-reload config)
            print("  ✓ Remote Spending allowed. Run /topup to continue.")
            return
        print("  ✓ Remote Spending allowed.")
        resume_choices = [
            ("resume", f"Resume {format_money(amount)} top-up", "finish the held purchase"),
            ("cancel", "Cancel", "do not charge")]
        if self._modal_choice("Resume your top-up", f"{format_money(amount)} is ready to finish — press Enter to resume.", resume_choices) != "resume":
            print("  Cancelled. No funds added.")
            return
        self._billing_submit_and_poll(
            fresh, amount, idempotency_key or new_idempotency_key(),
            missing_msg="  No charge id returned; please check the portal.",
            status_msg="Resuming your top-up — confirming settlement…")

    def _billing_auto_reload_flow(self, state):
        """Screen 4 — threshold + reload-to → PATCH. Prefills; validates ``reload_to > threshold``; "Turn off" if on."""
        from agent.billing_view import format_money, validate_charge_amount
        if not self._billing_require_admin(state):
            return
        card = state.card
        ar = state.auto_reload
        currently_on = bool(ar and ar.enabled)
        self._block_header("💳", "Auto-reload")
        self._dim('Automatically add funds when your balance is low.')
        if card:
            print(f"  Card on file: {card.masked}")
        else:
            print("  No saved card — manage billing on the portal.")
            self._billing_portal_hint(state)
            return
        _current = f"below {format_money(ar.threshold_usd)} → reload to {format_money(ar.reload_to_usd)}" if currently_on else ""
        if currently_on:
            print(f"  Currently: {_current}")
        if not self._app:
            print("  Run in the interactive CLI to configure auto-reload.")
            self._billing_portal_hint(state)
            return
        if currently_on:  # let the user turn it off without re-entering values
            top = self._modal_choice("Auto-reload", f"On — {_current}", _AUTO_RELOAD_TOP_CHOICES)
            if top == "off":
                self._billing_auto_reload_disable(state)
                return
            if top != "edit":
                print("  🟡 Cancelled.")
                return
        _CANCELLED = object()

        def _ask_amount(label, current):
            """One amount; empty keeps `current` when editing. Decimal / kept value, or _CANCELLED (already printed)."""
            cur = format_money(current) if currently_on else None
            raw = self._prompt_text_input(f"  {label} (USD)" + (f" [{cur}]: " if cur else ": "))
            if raw is None:  # cancelled (e.g. slash-worker can't prompt off-thread)
                print("  🟡 Cancelled.")
                return _CANCELLED
            if not (raw or "").strip() and currently_on:
                return current
            v = validate_charge_amount(raw or "", min_usd=state.min_usd, max_usd=state.max_usd)
            if not v.ok or v.amount is None:
                print(f"  🔴 {v.error}")
                return _CANCELLED
            return v.amount
        threshold_amt = _ask_amount("When balance falls below", ar.threshold_usd if currently_on else None)
        if threshold_amt is _CANCELLED:
            return
        reload_amt = _ask_amount("Reload balance to", ar.reload_to_usd if currently_on else None)
        if reload_amt is _CANCELLED:
            return
        if reload_amt is None or threshold_amt is None or reload_amt <= threshold_amt:
            print("  🔴 Reload-to amount must be greater than the threshold.")
            return
        self._dim(f"By confirming, you authorize Nous Research to charge {card.masked} whenever your balance reaches {format_money(threshold_amt)}. Turn off any time here or on the portal.", lead=True)
        if self._modal_choice("Turn on auto-reload?", f"Below {format_money(threshold_amt)} → reload to {format_money(reload_amt)}", _AUTO_RELOAD_AGREE_CHOICES) != "agree":
            print("  🟡 Cancelled.")
            return
        if self._billing_patch_auto_top_up(state, enabled=True, threshold=float(threshold_amt), top_up_amount=float(reload_amt)):
            print(f"  ✅ Auto-reload on: below {format_money(threshold_amt)} → reload to {format_money(reload_amt)}.")

    def _billing_patch_auto_top_up(self, state, **kwargs) -> bool:
        """PATCH auto-top-up; scope denials → step-up, other errors → renderer. True on success."""
        from hermes_cli.nous_billing import BillingError, BillingScopeRequired, patch_auto_top_up
        try:
            patch_auto_top_up(**kwargs)
        except BillingScopeRequired:
            self._billing_handle_scope_required(state)
            return False
        except BillingError as exc:
            self._billing_render_charge_error(state, exc)
            return False
        return True

    def _billing_auto_reload_disable(self, state):
        """PATCH ``enabled:false``; the endpoint still requires threshold/topUpAmount → echo current (or 0)."""
        ar = state.auto_reload
        thr = float(ar.threshold_usd) if ar and ar.threshold_usd is not None else 0.0
        rel = float(ar.reload_to_usd) if ar and ar.reload_to_usd is not None else 0.0
        if self._billing_patch_auto_top_up(state, enabled=False, threshold=thr, top_up_amount=rel):
            print("  ✅ Auto-reload turned off.")

    def _billing_limit_screen(self, state):
        """Screen 5 — monthly spend limit (read-only; cap is portal-only)."""
        from agent.billing_view import format_money
        self._block_header("💳", "Monthly spend limit")
        cap = state.monthly_cap
        if cap is None or cap.limit_usd is None:
            self._dim('No monthly cap visible (managed on the portal).')
        else:
            ceiling = " (default ceiling)" if cap.is_default_ceiling else ""
            print(f"  {format_money(cap.spent_this_month_usd)} of {format_money(cap.limit_usd)} used this month{ceiling}")
        self._dim('The monthly limit is set on the portal — the terminal shows it read-only.')
        self._billing_portal_hint(state)
