import { useState, type FormEvent, type KeyboardEvent } from "react";

interface ComposerProps {
  disabled: boolean;
  busy: boolean;
  promptSymbol: string;
  onSubmit: (text: string) => void;
  onInterrupt: () => void;
}

export function Composer({ disabled, busy, promptSymbol, onSubmit, onInterrupt }: ComposerProps) {
  const [value, setValue] = useState("");

  const send = () => {
    const text = value.trim();
    if (!text || disabled) return;
    onSubmit(text);
    setValue("");
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    send();
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter sends; Shift+Enter inserts a newline.
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <form className="ht-composer" onSubmit={handleSubmit}>
      <span className="ht-composer__symbol" aria-hidden>
        {promptSymbol}
      </span>
      <textarea
        className="ht-composer__input"
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={disabled ? "Connecting…" : "Message HT AI Agent…"}
        rows={1}
        disabled={disabled}
        aria-label="Message"
      />
      {busy ? (
        <button type="button" className="ht-btn ht-btn--stop" onClick={onInterrupt}>
          Stop
        </button>
      ) : (
        <button type="submit" className="ht-btn" disabled={disabled || value.trim().length === 0}>
          Send
        </button>
      )}
    </form>
  );
}
