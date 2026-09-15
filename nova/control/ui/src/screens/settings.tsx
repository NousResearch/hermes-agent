/* The tenant's own identity: who they are, and what their workforce is called.
 *
 * Both halves were editable only by editing YAML on the server. They are the first thing a
 * customer wants to change and the last thing anyone wants to hand them a text editor for.
 *
 * Branding here is not decoration. The accent is applied to the interface's own colour
 * tokens at boot, so changing it changes every surface — buttons, focus rings, the active
 * nav item — rather than one header. The preview below shows that before saving.
 *
 * The logo is uploaded into the bundle and served by the control plane from its own origin.
 * The Control Centre runs under `img-src 'self' data:`, so a hosted URL would be blocked by
 * the browser and appear as a broken image with no explanation.
 */

import * as React from "react";
import { Building2, Check, Loader2, Palette, Trash2, Upload } from "lucide-react";

import { GlassPanel, SectionHeader } from "@/components/glass";
import { TextArea, TextInput } from "@/components/form";
import { post } from "@/lib/api";
import { SaveBar, SaveResult, type SaveState, useEditable } from "@/lib/editing";
import { usePanel } from "@/lib/hooks";

type Settings = {
  organization: {
    tenant_id: string; legal_name: string; region: string;
    timezone: string; contact_email: string;
  };
  identity: {
    product_name: string; company_name: string;
    theme: { accent?: string; surface?: string; on_accent?: string };
    support: { email?: string; url?: string };
    messages: { welcome?: string; goodbye?: string };
    agents: Record<string, string>;
  };
  logo: { logo: boolean; favicon: boolean };
  settable: { organization: string[]; identity: string[] };
  immutable: string[];
};

export function SettingsScreen({ onChanged }: { onChanged?: () => void }) {
  const [nonce, setNonce] = React.useState(0);
  const loaded = usePanel<Settings>("/settings", 120000, nonce);
  const refresh = () => { setNonce((n) => n + 1); onChanged?.(); };

  if (loaded.state === "loading") {
    return <GlassPanel className="p-5"><p className="text-ink-faint text-[13px]">reading…</p></GlassPanel>;
  }
  if (loaded.state === "forbidden") {
    return <GlassPanel className="p-5"><p className="text-ink-muted text-[13px]">Not visible to your role.</p></GlassPanel>;
  }
  if (loaded.state === "error") {
    return (
      <GlassPanel className="p-5">
        <p className="text-blocked text-[13px]"><b>Settings could not be read.</b> {loaded.message}</p>
      </GlassPanel>
    );
  }

  return (
    <div className="space-y-5">
      <Company settings={loaded.data} onSaved={refresh} />
      <Branding settings={loaded.data} onSaved={refresh} />
      <LogoPanel settings={loaded.data} onSaved={refresh} />
    </div>
  );
}

/* ── Company ──────────────────────────────────────────────────────────────── */

function Company({ settings, onSaved }: { settings: Settings; onSaved: () => void }) {
  const server = React.useMemo(() => ({
    legal_name: settings.organization.legal_name ?? "",
    region: settings.organization.region ?? "",
    timezone: settings.organization.timezone ?? "",
    contact_email: settings.organization.contact_email ?? "",
  }), [settings]);

  const editor = useEditable(server, async (value) => {
    const result = await post("/settings/organization", { fields: value });
    onSaved();
    return result;
  });
  const v = editor.value;

  return (
    <GlassPanel className="p-5">
      <SectionHeader
        icon={Building2} title="Company"
        detail="Who this deployment serves. Recorded against the work it does."
      />
      <div className="grid gap-4 sm:grid-cols-2">
        <TextInput id="org-legal" label="Legal name" value={v.legal_name}
          onChange={(legal_name) => editor.edit((c) => ({ ...c, legal_name }))}
          hint="The registered name, as it should appear on anything official." />
        <TextInput id="org-contact" label="Contact email" value={v.contact_email}
          onChange={(contact_email) => editor.edit((c) => ({ ...c, contact_email }))}
          hint="Where an operator is reached about this deployment." />
        <TextInput id="org-region" label="Region" value={v.region} mono
          onChange={(region) => editor.edit((c) => ({ ...c, region }))}
          hint="Where this tenant runs. Used when rendering infrastructure." />
        <TextInput id="org-tz" label="Timezone" value={v.timezone} mono
          onChange={(timezone) => editor.edit((c) => ({ ...c, timezone }))}
          hint="How schedules are read. Europe/London, America/New_York." />
      </div>

      <div className="border-glass-border mt-4 rounded-lg border p-3">
        <p className="text-ink-faint text-[11px] font-medium tracking-wide uppercase">Tenant id</p>
        <p className="text-ink mt-1 font-mono text-[12.5px]">{settings.organization.tenant_id}</p>
        <p className="text-ink-faint mt-1.5 text-[11.5px] leading-relaxed">
          Fixed. It is stamped on every audit event already recorded and names this tenant's
          runtime home, so changing it would orphan the history rather than rename the tenant.
        </p>
      </div>

      <SaveBar dirty={editor.dirty} busy={editor.busy} state={editor.state}
               onSave={editor.submit} onDiscard={editor.discard} label="Save company" />
    </GlassPanel>
  );
}

/* ── Branding ─────────────────────────────────────────────────────────────── */

const SWATCHES = ["#1F6F5C", "#0F62FE", "#7C3AED", "#B91C1C", "#B45309", "#0F766E", "#1E293B"];

function Branding({ settings, onSaved }: { settings: Settings; onSaved: () => void }) {
  const server = React.useMemo(() => ({
    product_name: settings.identity.product_name ?? "",
    company_name: settings.identity.company_name ?? "",
    theme: {
      accent: settings.identity.theme?.accent ?? "",
      on_accent: settings.identity.theme?.on_accent ?? "",
    },
    support: {
      email: settings.identity.support?.email ?? "",
      url: settings.identity.support?.url ?? "",
    },
    messages: {
      welcome: settings.identity.messages?.welcome ?? "",
      goodbye: settings.identity.messages?.goodbye ?? "",
    },
  }), [settings]);

  const editor = useEditable(server, async (value) => {
    const result = await post("/settings/identity", { fields: value });
    onSaved();
    return result;
  });
  const v = editor.value;
  const accent = v.theme.accent || "var(--accent)";
  const onAccent = v.theme.on_accent || "var(--accent-ink)";

  return (
    <GlassPanel className="p-5">
      <SectionHeader
        icon={Palette} title="Brand"
        detail="What the workforce is called, and the colour it wears everywhere."
      />

      <div className="grid gap-4 sm:grid-cols-2">
        <TextInput id="brand-product" label="Product name" value={v.product_name}
          onChange={(product_name) => editor.edit((c) => ({ ...c, product_name }))}
          hint="What this is called on every screen, in the tab title and in the agents' own introductions." />
        <TextInput id="brand-company" label="Company name" value={v.company_name}
          onChange={(company_name) => editor.edit((c) => ({ ...c, company_name }))}
          hint="Shown in the footer and wherever the product names its owner." />
      </div>

      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        <div className="space-y-1.5">
          <label htmlFor="brand-accent" className="text-ink block text-[12.5px] font-medium">
            Accent colour
          </label>
          <div className="flex items-center gap-2">
            <input
              id="brand-accent" type="color"
              value={/^#[0-9a-fA-F]{6}$/.test(v.theme.accent) ? v.theme.accent : "#1f6f5c"}
              onChange={(e) => editor.edit((c) => ({ ...c, theme: { ...c.theme, accent: e.target.value } }))}
              className="border-glass-border size-9 shrink-0 cursor-pointer rounded-lg border bg-transparent p-1"
            />
            <input
              type="text" value={v.theme.accent} placeholder="#1F6F5C" spellCheck={false}
              onChange={(e) => editor.edit((c) => ({ ...c, theme: { ...c.theme, accent: e.target.value } }))}
              className="border-glass-border bg-glass text-ink focus-visible:ring-info/50 min-w-0 flex-1 rounded-lg border px-3 py-2 font-mono text-[12.5px] outline-none focus-visible:ring-2"
            />
          </div>
          <div className="flex flex-wrap gap-1.5 pt-1">
            {SWATCHES.map((hex) => (
              <button
                key={hex} type="button" title={hex}
                aria-label={`Use ${hex}`}
                onClick={() => editor.edit((c) => ({ ...c, theme: { ...c.theme, accent: hex } }))}
                className="border-glass-border size-6 rounded-md border"
                style={{ background: hex }}
              />
            ))}
          </div>
          <p className="text-ink-faint text-[11.5px] leading-relaxed">
            Applied to the interface's own colour tokens, so it reaches every surface rather
            than one header. Leave blank for NOVA's default.
          </p>
        </div>

        <div className="space-y-1.5">
          <span className="text-ink block text-[12.5px] font-medium">Preview</span>
          <div className="border-glass-border rounded-lg border p-3">
            <div className="flex items-center gap-2.5">
              <div className="grid size-8 shrink-0 place-items-center rounded-lg text-[13px] font-bold"
                   style={{ background: accent, color: onAccent }}>
                {(v.product_name || "N").slice(0, 1).toUpperCase()}
              </div>
              <div className="min-w-0">
                <div className="text-ink truncate text-[13px] font-semibold">
                  {v.product_name || "Control Center"}
                </div>
                <div className="text-ink-faint truncate font-mono text-[10px]">
                  {settings.organization.tenant_id}
                </div>
              </div>
            </div>
            <button type="button" disabled
              className="mt-3 w-full rounded-lg px-3 py-1.5 text-[12px] font-medium"
              style={{ background: accent, color: onAccent }}>
              A button in this brand
            </button>
          </div>
          <p className="text-ink-faint text-[11.5px]">
            Saving applies it to the whole interface, not just this box.
          </p>
        </div>
      </div>

      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        <TextInput id="brand-support-email" label="Support email" value={v.support.email}
          onChange={(email) => editor.edit((c) => ({ ...c, support: { ...c.support, email } }))}
          hint="Where a user of the branded product goes for a human." />
        <TextInput id="brand-support-url" label="Support link" value={v.support.url}
          onChange={(url) => editor.edit((c) => ({ ...c, support: { ...c.support, url } }))} />
      </div>

      <div className="mt-4">
        <TextArea id="brand-welcome" label="Welcome message" value={v.messages.welcome} rows={2}
          onChange={(welcome) => editor.edit((c) => ({ ...c, messages: { ...c.messages, welcome } }))}
          hint="What a customer sees first when an agent greets them." />
      </div>

      <SaveBar dirty={editor.dirty} busy={editor.busy} state={editor.state}
               onSave={editor.submit} onDiscard={editor.discard} label="Save brand" />
    </GlassPanel>
  );
}

/* ── Logo ─────────────────────────────────────────────────────────────────── */

const ACCEPTED = "image/png,image/jpeg,image/webp,image/gif";

function LogoPanel({ settings, onSaved }: { settings: Settings; onSaved: () => void }) {
  const [state, setState] = React.useState<SaveState>({ kind: "idle" });
  const [pending, setPending] = React.useState<string | null>(null);
  // Bumped after an upload so the browser re-fetches an image at an unchanged URL.
  const [version, setVersion] = React.useState(0);

  async function upload(kind: "logo" | "favicon", file: File) {
    if (pending) return;
    setPending(kind);
    setState({ kind: "saving" });
    try {
      const data: string = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result ?? ""));
        reader.onerror = () => reject(new Error("the file could not be read"));
        reader.readAsDataURL(file);
      });
      const result: any = await post("/settings/logo", {
        kind, data, content_type: file.type,
      });
      setState({
        kind: "saved", applied: Boolean(result?.runtime?.applied ?? true),
        detail: String(result?.runtime?.error ?? ""), files: result?.files_changed ?? [],
      });
      setVersion((n) => n + 1);
      onSaved();
    } catch (cause) {
      setState({
        kind: "error",
        message: cause instanceof Error ? cause.message : "the upload did not complete",
      });
    } finally {
      setPending(null);
    }
  }

  async function clear(kind: "logo" | "favicon") {
    if (pending) return;
    setPending(kind);
    setState({ kind: "saving" });
    try {
      const result: any = await post("/settings/logo", { kind, data: "" });
      setState({ kind: "saved", applied: true, detail: "", files: result?.files_changed ?? [] });
      setVersion((n) => n + 1);
      onSaved();
    } catch (cause) {
      setState({
        kind: "error",
        message: cause instanceof Error ? cause.message : "the request did not complete",
      });
    } finally {
      setPending(null);
    }
  }

  return (
    <GlassPanel className="p-5">
      <SectionHeader
        title="Logo" detail="Shown beside the product name, and as the browser tab icon."
      />
      <div className="grid gap-4 sm:grid-cols-2">
        {(["logo", "favicon"] as const).map((kind) => (
          <div key={kind} className="border-glass-border rounded-lg border p-3">
            <div className="flex items-center gap-3">
              <div className="border-glass-border grid size-12 shrink-0 place-items-center overflow-hidden rounded-lg border">
                {settings.logo[kind] ? (
                  <img
                    src={`/platform/v1/branding/${kind}?v=${version}`}
                    alt={`${kind} preview`}
                    className="size-full object-contain"
                  />
                ) : (
                  <span className="text-ink-faint text-[10px]">none</span>
                )}
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-ink text-[12.5px] font-medium capitalize">{kind}</p>
                <p className="text-ink-faint mt-0.5 text-[11.5px]">
                  {settings.logo[kind] ? "Stored in this tenant's bundle." : "Not set."}
                </p>
              </div>
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <label className="glass-solid text-ink inline-flex cursor-pointer items-center gap-1.5 rounded-lg px-3 py-1.5 text-[12px] font-medium">
                {pending === kind ? <Loader2 className="size-3.5 animate-spin" /> : <Upload className="size-3.5" />}
                {pending === kind ? "Uploading…" : settings.logo[kind] ? "Replace" : "Upload"}
                <input
                  type="file" accept={ACCEPTED} className="sr-only"
                  disabled={pending !== null}
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    e.target.value = "";   // so choosing the same file twice still fires
                    if (file) void upload(kind, file);
                  }}
                />
              </label>
              {settings.logo[kind] ? (
                <button
                  type="button" onClick={() => void clear(kind)} disabled={pending !== null}
                  className="border-glass-border text-ink-muted hover:text-ink inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-[12px] disabled:opacity-40"
                >
                  <Trash2 className="size-3.5" /> Remove
                </button>
              ) : null}
            </div>
          </div>
        ))}
      </div>

      <p className="text-ink-faint mt-3 text-[11.5px] leading-relaxed">
        PNG, JPEG, WebP or GIF, up to 1.5 MB. SVG is not accepted: it can carry script, and
        this image is served from the control plane's own origin. The file is stored in the
        tenant's bundle and served from there — a hosted URL would be blocked by the
        interface's content security policy and show as a broken image.
      </p>

      <SaveResult state={state} />
      {state.kind === "saved" ? (
        <p className="text-ink-faint mt-2 flex items-center gap-1.5 text-[11.5px]">
          <Check className="size-3" /> The sidebar updates on the next read.
        </p>
      ) : null}
    </GlassPanel>
  );
}
