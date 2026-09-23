import {
  CircleCheck, Code, FileText, Image, MessagesSquare, Mic, PenLine, Radio, Trash2, Type,
  Users, Video,
} from "lucide-react";
import { Hint } from "@/components/tooltip";
import type { Capability } from "@/screens/types";

/* What a channel can actually do, as read from the runtime's adapter.
 *
 * Only supported capabilities get a chip; the rest collapse into one "not native" note
 * with each fallback named in its tooltip, because what a customer on that channel gets
 * instead ("the image URL as text") is the useful part. Unknown is never drawn as yes. */

const META: Record<string, { label: string; icon: typeof Image }> = {
  images: { label: "Images", icon: Image },
  documents: { label: "Files", icon: FileText },
  voice: { label: "Voice", icon: Mic },
  video: { label: "Video", icon: Video },
  message_edits: { label: "Live edits", icon: PenLine },
  draft_streaming: { label: "Streaming", icon: Radio },
  message_delete: { label: "Delete", icon: Trash2 },
  typing_indicator: { label: "Typing", icon: Type },
  approval_buttons: { label: "Approval buttons", icon: CircleCheck },
  code_blocks: { label: "Code blocks", icon: Code },
  threads: { label: "Threads", icon: MessagesSquare },
  groups: { label: "Groups", icon: Users },
};

export function CapabilityList({ capabilities }: { capabilities?: Record<string, Capability> }) {
  const entries = Object.entries(capabilities ?? {}).filter(([key]) => key in META);
  if (!entries.length) return null;
  const yes = entries.filter(([, cap]) => cap.supported === true);
  const no = entries.filter(([, cap]) => cap.supported === false);

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {yes.map(([key, cap]) => {
        const { label, icon: Icon } = META[key];
        return (
          <Hint key={key} text={cap.note || `${label}: supported natively`}>
            <span className="border-glass-border bg-glass-1 text-ink-muted inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[11.5px]">
              <Icon className="size-3" aria-hidden /> {label}
            </span>
          </Hint>
        );
      })}
      {no.length ? (
        <Hint text={no.map(([key, cap]) => `${META[key].label}: ${cap.note || "not native"}`).join(" · ")}>
          <span className="text-ink-faint text-[11.5px] underline decoration-dotted underline-offset-2">
            {no.length} not native
          </span>
        </Hint>
      ) : null}
    </div>
  );
}
