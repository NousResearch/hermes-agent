/* Form controls, in the Control Centre's own idiom.
 *
 * Plain inputs styled from the existing tokens rather than a component library: the page
 * already has a design system, and adding a second one for six forms would mean two
 * definitions of what a focus ring looks like.
 *
 * Every control takes a stable `id` and renders a real `<label>` bound to it — the platform
 * carries form values, focus and scroll across a republish, and a screen reader needs the
 * association regardless.
 */

import * as React from "react";
import { Check } from "lucide-react";

const INPUT =
  "border-glass-border bg-glass text-ink focus-visible:ring-info/50 w-full rounded-lg border " +
  "px-3 py-2 text-[13px] outline-none transition-shadow focus-visible:ring-2 " +
  "disabled:opacity-50";

export function Field({
  id, label, hint, error, children,
}: {
  id: string; label: string; hint?: React.ReactNode; error?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5">
      <label htmlFor={id} className="text-ink block text-[12.5px] font-medium">
        {label}
      </label>
      {children}
      {error ? (
        <p className="text-blocked text-[11.5px]">{error}</p>
      ) : hint ? (
        <p className="text-ink-faint text-[11.5px] leading-relaxed">{hint}</p>
      ) : null}
    </div>
  );
}

export function TextInput({
  id, value, onChange, placeholder, hint, label, error, mono = false, disabled = false,
}: {
  id: string; value: string; onChange: (v: string) => void;
  placeholder?: string; hint?: React.ReactNode; label: string; error?: string;
  mono?: boolean; disabled?: boolean;
}) {
  return (
    <Field id={id} label={label} hint={hint} error={error}>
      <input
        id={id} type="text" value={value} placeholder={placeholder} disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
        className={`${INPUT} ${mono ? "font-mono text-[12.5px]" : ""} ${
          error ? "border-blocked/50" : ""
        }`}
      />
    </Field>
  );
}

export function TextArea({
  id, value, onChange, label, hint, rows = 4, mono = false, placeholder,
}: {
  id: string; value: string; onChange: (v: string) => void; label: string;
  hint?: React.ReactNode; rows?: number; mono?: boolean; placeholder?: string;
}) {
  return (
    <Field id={id} label={label} hint={hint}>
      <textarea
        id={id} value={value} rows={rows} placeholder={placeholder} spellCheck={!mono}
        onChange={(e) => onChange(e.target.value)}
        className={`${INPUT} resize-y leading-relaxed ${mono ? "font-mono text-[12.5px]" : ""}`}
      />
    </Field>
  );
}

export function Toggle({
  id, checked, onChange, label, hint,
}: {
  id: string; checked: boolean; onChange: (v: boolean) => void; label: string;
  hint?: React.ReactNode;
}) {
  return (
    <div className="flex items-start gap-2.5">
      <input
        id={id} type="checkbox" checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="accent-info mt-0.5 size-4 shrink-0"
      />
      <div>
        <label htmlFor={id} className="text-ink block text-[12.5px] font-medium">
          {label}
        </label>
        {hint ? <p className="text-ink-faint mt-0.5 text-[11.5px]">{hint}</p> : null}
      </div>
    </div>
  );
}

/** Multi-select over a known set. Nothing may be typed that is not offered — the options
 *  come from the tenant's policy or the runtime's registry, and a free-text field would
 *  let somebody name a permission that does not exist and learn about it at save time. */
export function ChipSelect({
  label, hint, options, selected, onChange, emptyNote, describe,
}: {
  label: string; hint?: React.ReactNode; options: string[]; selected: string[];
  onChange: (next: string[]) => void; emptyNote?: string;
  describe?: (id: string) => string | undefined;
}) {
  const chosen = new Set(selected);
  return (
    <div className="space-y-1.5">
      <span className="text-ink block text-[12.5px] font-medium">{label}</span>
      {hint ? <p className="text-ink-faint text-[11.5px] leading-relaxed">{hint}</p> : null}
      {options.length === 0 ? (
        <p className="text-ink-faint text-[12px]">{emptyNote ?? "Nothing to choose from."}</p>
      ) : (
        <div className="flex flex-wrap gap-1.5 pt-0.5">
          {options.map((option) => {
            const on = chosen.has(option);
            return (
              <button
                key={option} type="button" aria-pressed={on}
                title={describe?.(option)}
                onClick={() =>
                  onChange(on ? selected.filter((s) => s !== option) : [...selected, option])
                }
                className={`inline-flex items-center gap-1 rounded-md border px-2 py-1
                            font-mono text-[11.5px] transition-colors ${
                  on
                    ? "border-running/40 bg-running/10 text-running"
                    : "border-glass-border text-ink-muted hover:text-ink"
                }`}
              >
                {on ? <Check className="size-3" /> : null}
                {option}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

export function Select({
  id, label, value, onChange, options, hint,
}: {
  id: string; label: string; value: string; onChange: (v: string) => void;
  options: { value: string; label: string }[]; hint?: React.ReactNode;
}) {
  return (
    <Field id={id} label={label} hint={hint}>
      <select
        id={id} value={value} onChange={(e) => onChange(e.target.value)}
        className={INPUT}
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </Field>
  );
}
