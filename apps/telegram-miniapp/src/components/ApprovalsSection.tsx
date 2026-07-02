import { useState } from "react";
import { type ApprovalDecision } from "../api";
import { type ApprovalDecisionValue, type ApprovalPreview } from "../mockData";
import { EmptyState, RiskBadge } from "./chrome";

const DECISION_LABEL: Record<ApprovalDecisionValue, string> = {
  approve_once: "Одобрить один раз",
  reject_once: "Отклонить один раз",
};

function DecisionReadiness({ approval, live }: { approval: ApprovalPreview; live: boolean }) {
  const guardrails = [
    {
      label: "Действия сервера",
      value: live ? "подключены" : "не подключены",
      ready: live,
      detail: live
        ? "Approve/reject once доступны только владельцу через отдельное подтверждение."
        : "В Mini App нет endpoint для approve/reject/restart. Их добавление требует отдельного approved design.",
    },
    {
      label: "Контекст владельца",
      value: live ? "подтверждён" : approval.risk === "critical" ? "требуется" : "ожидает",
      ready: live,
      detail: "Решение требует свежего Telegram-подтверждения владельца и второго шага.",
    },
    {
      label: "Проверки запроса",
      value: `${approval.checks.length} видно`,
      ready: approval.checks.length > 0,
      detail: "Список отображается как read-only preflight; сырая команда не раскрывается.",
    },
  ] as const;

  return (
    <section className="decision-readiness" aria-label="Готовность решения">
      <div className="decision-readiness-head">
        <div>
          <p className="mono-label">{live ? "M18 / ACTION GATE" : "M12 / PREFLIGHT LOCK"}</p>
          <h3>{live ? "Решение владельца в два шага" : "Решение требует отдельного design approval"}</h3>
        </div>
        <span className="lock-pill">{live ? "owner" : "read-only"}</span>
      </div>
      <p>
        {live
          ? "Approve/reject once отправляется только после подтверждения. Сырая команда в Mini App не показывается; gateway перепроверяет решение."
          : "Этот блок заранее показывает, что должно быть проверено перед будущим approve/reject. Сейчас он ничего не отправляет, не меняет и не создаёт action route."}
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
  onConfirm,
  onCancel,
}: {
  approval: ApprovalPreview;
  decision: ApprovalDecisionValue;
  pending: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <div className="sheet-backdrop" role="presentation" onClick={pending ? undefined : onCancel}>
      <section
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
          <small>Mini App показывает только редактированное описание; полную команду видит и исполняет gateway после проверки.</small>
        </div>
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
  isOwner = false,
  onDecision,
}: {
  approvals: ApprovalPreview[];
  selectedId: string;
  onSelect: (approval: ApprovalPreview) => void;
  actionsEnabled?: boolean;
  isOwner?: boolean;
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
  // A decision is actionable only when the backend capability is live, the user
  // is the authenticated owner, and this approval advertises the decision.
  const canDecide = actionsEnabled && isOwner && Boolean(onDecision);

  async function runDecision(target: ApprovalPreview, decision: ApprovalDecisionValue) {
    if (!onDecision) return;
    setPending(true);
    try {
      await onDecision(target.id, decision);
    } finally {
      setPending(false);
      setConfirm(null);
    }
  }

  return (
    <section className="approval-workspace" aria-label="Очередь одобрений">
      <div className="stack-list compact-stack">
        {approvals.map((approval) => (
          <button className="approval-row glass-card tap" data-selected={approval.id === selected.id} key={approval.id} type="button" onClick={() => onSelect(approval)}>
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
        <DecisionReadiness approval={selected} live={canDecide} />
        <div className="decision-strip" aria-label="Решения">
          <button
            type="button"
            className="tap"
            disabled={!canDecide || pending || !allowed.includes("approve_once")}
            onClick={() => setConfirm({ approval: selected, decision: "approve_once" })}
          >
            {canDecide ? "Одобрить" : "Одобрить позже"}
          </button>
          <button
            type="button"
            className="tap"
            disabled={!canDecide || pending || !allowed.includes("reject_once")}
            onClick={() => setConfirm({ approval: selected, decision: "reject_once" })}
          >
            {canDecide ? "Отклонить" : "Отклонить позже"}
          </button>
        </div>
      </article>

      {confirm ? (
        <ConfirmSheet
          approval={confirm.approval}
          decision={confirm.decision}
          pending={pending}
          onConfirm={() => void runDecision(confirm.approval, confirm.decision)}
          onCancel={() => setConfirm(null)}
        />
      ) : null}
    </section>
  );
}
