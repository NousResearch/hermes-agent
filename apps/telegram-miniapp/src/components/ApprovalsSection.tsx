import { useState, type RefObject } from "react";
import { type ApprovalDecision } from "../api";
import { type ApprovalDecisionValue, type ApprovalPreview } from "../mockData";
import { EmptyState, RiskBadge } from "./chrome";
import { useDialog } from "./useDialog";
import { triggerTelegramImpact } from "../telegram";

const DECISION_LABEL: Record<ApprovalDecisionValue, string> = {
  approve_once: "Одобрить один раз",
  reject_once: "Отклонить один раз",
};

function DecisionReadiness({ approval, canSubmit, applicationEnabled }: { approval: ApprovalPreview; canSubmit: boolean; applicationEnabled: boolean }) {
  const guardrails = [
    {
      label: "Контур решений",
      value: canSubmit ? "можно записать" : applicationEnabled ? "ждёт свежий снимок" : "не подключён",
      ready: canSubmit,
      detail: canSubmit
        ? "Sidecar примет решение владельца и запишет его в record-only bridge."
        : applicationEnabled
          ? "Кнопки закрыты, пока нет свежей версии очереди и подтверждённого владельца."
          : "Нет live-применения gateway; без отдельного resolver loop решения не исполняются.",
    },
    {
      label: "Контекст владельца",
      value: canSubmit ? "подтверждён" : approval.risk === "critical" ? "требуется" : "ожидает",
      ready: canSubmit,
      detail: "Решение требует свежего Telegram-подтверждения владельца и второго шага.",
    },
    {
      label: "Проверки запроса",
      value: `${approval.checks.length} видно`,
      ready: approval.checks.length > 0,
      detail: "Список только для чтения; сырая команда не раскрывается.",
    },
  ] as const;

  return (
    <section className="decision-readiness" aria-label="Готовность решения">
      <div className="decision-readiness-head">
        <div>
          <p className="mono-label">{canSubmit ? "M18 / RECORD-ONLY GATE" : "M12 / PREFLIGHT LOCK"}</p>
          <h3>{canSubmit ? "Решение владельца записывается в два шага" : "Решение пока недоступно"}</h3>
        </div>
        <span className="lock-pill">{canSubmit ? "владелец" : "закрыто"}</span>
      </div>
      <p>
        {canSubmit
          ? "Решение «одобрить/отклонить один раз» записывается только после подтверждения. Сырая команда в мини-аппе не показывается. Это record-only: gateway пока не применяет решение автоматически."
          : "Этот блок показывает, что должно быть свежим перед решением. Пока кнопки закрыты: без свежего снимка, владельца и record-only gate ничего не отправляется."}
      </p>
      <div className="readiness-grid">
        {guardrails.map((item) => (
          <article className="readiness-item" data-ready={item.ready} key={item.label}>
            <span>{item.label}</span>
            <strong>{item.value}</strong>
            <small>{item.detail}</small>
          </article>
        ))}
      </div>
    </section>
  );
}

function ConfirmSheet({
  approval,
  decision,
  pending,
  error,
  onConfirm,
  onCancel,
}: {
  approval: ApprovalPreview;
  decision: ApprovalDecisionValue;
  pending: boolean;
  error?: string;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  // Trap focus in the destructive-action dialog, close on Escape (unless
  // sending), and restore focus to the decision button on close.
  const dialogRef = useDialog(true, onCancel, { busy: pending });
  return (
    <div className="sheet-backdrop" role="presentation" onClick={pending ? undefined : onCancel}>
      <section
        ref={dialogRef as RefObject<HTMLElement>}
        className="command-sheet glass-card"
        aria-label="Подтверждение решения"
        role="dialog"
        aria-modal="true"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="sheet-handle" />
        <div className="section-heading compact">
          <div>
            <p className="mono-label">ШАГ 2 · ПОДТВЕРЖДЕНИЕ</p>
            <h2>{DECISION_LABEL[decision]}</h2>
          </div>
          <RiskBadge action={approval} />
        </div>
        <p className="muted">{approval.title}</p>
        <div className="readonly-note">
          <strong>Сырая команда не раскрывается</strong>
          <small>Mini App показывает только редактированное описание; gateway не применяет это решение автоматически, пока resolver loop не подключён.</small>
        </div>
        {error ? <p className="decision-feedback error" role="alert">{error}</p> : null}
        <div className="decision-strip" aria-label="Подтвердить или отменить">
          <button type="button" className="tap" onClick={onCancel} disabled={pending}>
            Отмена
          </button>
          <button type="button" className="tap confirm-danger" onClick={onConfirm} disabled={pending}>
            {pending ? "Отправляю…" : DECISION_LABEL[decision]}
          </button>
        </div>
      </section>
    </div>
  );
}

export function ApprovalsSection({
  approvals,
  selectedId,
  onSelect,
  actionsEnabled = false,
  canSubmitDecision = false,
  decisionBlockReason = "",
  decisionError = "",
  onClearDecisionError,
  isOwner = false,
  isConnected = false,
  onDecision,
}: {
  approvals: ApprovalPreview[];
  selectedId: string;
  onSelect: (approval: ApprovalPreview) => void;
  actionsEnabled?: boolean;
  canSubmitDecision?: boolean;
  decisionBlockReason?: string;
  decisionError?: string;
  onClearDecisionError?: () => void;
  isOwner?: boolean;
  isConnected?: boolean;
  onDecision?: (approvalId: string, decision: ApprovalDecision) => Promise<boolean>;
}) {
  // The confirm freezes BOTH the decision and the exact approval it targets, so
  // a poll-refresh or a selection change between step 1 and step 2 can never
  // redirect the decision onto a different pending approval.
  const [confirm, setConfirm] = useState<{ approval: ApprovalPreview; decision: ApprovalDecisionValue } | null>(null);
  const [pending, setPending] = useState(false);

  if (approvals.length === 0) {
    return <EmptyState title="Очередь одобрений пуста" text="Сервер вернул пустую очередь. Это считается валидным live-состоянием, а не поводом показывать mock-запросы." />;
  }

  const selected = approvals.find((approval) => approval.id === selectedId) ?? approvals[0];
  const allowed = selected.allowedDecisions ?? [];
  // A decision is actionable only when hook-level readiness includes capability,
  // owner proof, connected state, record-only/live-honest application state and
  // a fresh approvals snapshot_version.
  const canDecide = canSubmitDecision && Boolean(onDecision);
  // Distinguish WHY a decision is unavailable so the button copy tells the
  // truth instead of a generic "позже": a non-owner on a live connection can
  // never decide (stable), whereas an owner whose gate is off/degraded may be
  // able to later.
  const denyReason: "none" | "not-owner" | "gate-off" | "preview" = canDecide
    ? "none"
    : isConnected && !isOwner
      ? "not-owner"
      : isConnected
        ? "gate-off"
        : "preview";
  const approveLabel = canDecide ? "Одобрить" : denyReason === "not-owner" ? "Только владелец" : "Одобрить позже";
  const rejectLabel = canDecide ? "Отклонить" : denyReason === "not-owner" ? "Только владелец" : "Отклонить позже";

  async function runDecision(target: ApprovalPreview, decision: ApprovalDecisionValue) {
    if (!onDecision) return;
    // Tactile confirmation at the moment the owner commits a destructive action.
    triggerTelegramImpact("rigid");
    setPending(true);
    try {
      const ok = await onDecision(target.id, decision);
      if (ok) setConfirm(null);
    } finally {
      setPending(false);
    }
  }

  function openConfirm(decision: ApprovalDecisionValue) {
    if (!canDecide) return;
    onClearDecisionError?.();
    setConfirm({ approval: selected, decision });
  }

  return (
    <section className="approval-workspace" aria-label="Очередь одобрений">
      <div className="stack-list compact-stack">
        {approvals.map((approval) => (
          <button className="approval-row glass-card tap" data-selected={approval.id === selected.id} aria-pressed={approval.id === selected.id} key={approval.id} type="button" onClick={() => onSelect(approval)}>
            <span>
              <strong>{approval.title}</strong>
              <small>{approval.source}</small>
            </span>
            <RiskBadge action={approval} />
          </button>
        ))}
      </div>

      <article className="approval-detail glass-card">
        <div className="section-heading compact">
          <div>
            <p className="mono-label">ДЕТАЛЬ ЗАПРОСА</p>
            <h2>{selected.title}</h2>
          </div>
          <RiskBadge action={selected} />
        </div>
        <p>{selected.summary}</p>
        <div className="detail-meta">
          <span>{selected.status}</span>
          <span>{selected.requestedAt}</span>
        </div>
        <ul className="check-list">
          {selected.checks.map((check) => (
            <li key={check}>{check}</li>
          ))}
        </ul>
        <DecisionReadiness approval={selected} canSubmit={canDecide} applicationEnabled={actionsEnabled} />
        {!canDecide && decisionBlockReason ? (
          <p className="decision-feedback" role="status">{decisionBlockReason}</p>
        ) : null}
        {decisionError ? (
          <p className="decision-feedback error" role="alert">{decisionError}</p>
        ) : null}
        <div className="decision-strip" aria-label="Решения">
          <button
            type="button"
            className="tap"
            disabled={!canDecide || pending || !allowed.includes("approve_once")}
            onClick={() => openConfirm("approve_once")}
          >
            {approveLabel}
          </button>
          <button
            type="button"
            className="tap"
            disabled={!canDecide || pending || !allowed.includes("reject_once")}
            onClick={() => openConfirm("reject_once")}
          >
            {rejectLabel}
          </button>
        </div>
      </article>

      {confirm ? (
        <ConfirmSheet
          approval={confirm.approval}
          decision={confirm.decision}
          pending={pending}
          error={decisionError}
          onConfirm={() => void runDecision(confirm.approval, confirm.decision)}
          onCancel={() => setConfirm(null)}
        />
      ) : null}
    </section>
  );
}
