import { useState } from "react";
import type { PendingApproval, PendingClarify } from "@/gateway/chatReducer";

export function ClarifyDialog({
  clarify,
  onRespond,
}: {
  clarify: PendingClarify;
  onRespond: (answer: string) => void;
}) {
  const [text, setText] = useState("");
  return (
    <div className="ht-dialog" role="dialog" aria-label="Clarification requested">
      <p className="ht-dialog__q">{clarify.question}</p>
      {clarify.choices && clarify.choices.length > 0 ? (
        <div className="ht-dialog__choices">
          {clarify.choices.map((c) => (
            <button key={c} type="button" className="ht-btn" onClick={() => onRespond(c)}>
              {c}
            </button>
          ))}
        </div>
      ) : (
        <form
          className="ht-dialog__form"
          onSubmit={(e) => {
            e.preventDefault();
            if (text.trim()) onRespond(text.trim());
          }}
        >
          <input
            className="ht-composer__input"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Your answer…"
            aria-label="Clarification answer"
            autoFocus
          />
          <button type="submit" className="ht-btn" disabled={!text.trim()}>
            Answer
          </button>
        </form>
      )}
    </div>
  );
}

export function ApprovalDialog({
  approval,
  onRespond,
}: {
  approval: PendingApproval;
  onRespond: (choice: string, all?: boolean) => void;
}) {
  return (
    <div className="ht-dialog ht-dialog--approval" role="dialog" aria-label="Approval requested">
      <p className="ht-dialog__q">{approval.description}</p>
      <pre className="ht-dialog__cmd">{approval.command}</pre>
      <div className="ht-dialog__choices">
        <button type="button" className="ht-btn" onClick={() => onRespond("allow")}>
          Allow
        </button>
        {approval.allowPermanent && (
          <button type="button" className="ht-btn" onClick={() => onRespond("allow", true)}>
            Always allow
          </button>
        )}
        <button type="button" className="ht-btn ht-btn--stop" onClick={() => onRespond("deny")}>
          Deny
        </button>
      </div>
    </div>
  );
}
