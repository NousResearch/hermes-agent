// @vitest-environment jsdom
import { AssistantRuntimeProvider, useExternalStoreRuntime } from "@assistant-ui/react";
import type { ThreadMessageLike } from "@assistant-ui/react";
import { act, cleanup, fireEvent, render } from "@testing-library/react";
import { MemoryRouter } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";

import { I18nProvider } from "@/i18n";
import { mainComposerScope } from "@/store/composer";

import { composerPlainText, RICH_INPUT_SLOT } from "./rich-editor";
import type { ChatBarState } from "./types";

import { ChatBar } from "./index";

afterEach(cleanup);

// THE INVARIANT (#128823): pasting OS files attaches them instead of dumping
// the clipboard text form into the composer. See the describe block below.
const state: ChatBarState = {
  model: { canSwitch: false, model: "", provider: "" },
  tools: { enabled: false, label: "" },
  voice: { enabled: false, active: false }
};

function Harness({ onAttachDroppedItems }: { onAttachDroppedItems: (candidates: unknown[]) => Promise<boolean> }) {
  const runtime = useExternalStoreRuntime({
    convertMessage: (message: ThreadMessageLike) => message,
    isRunning: false,
    messages: [] as ThreadMessageLike[],
    onNew: async () => {}
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <MemoryRouter>
        <I18nProvider configClient={null} initialLocale="en">
          <ChatBar
            busy={false}
            disabled={false}
            gateway={null}
            onAttachDroppedItems={onAttachDroppedItems as never}
            onCancel={vi.fn()}
            onSubmit={vi.fn(async () => true)}
            state={state}
          />
        </I18nProvider>
      </MemoryRouter>
    </AssistantRuntimeProvider>
  );
}

function editorSlot(): string {
  return "[data-slot=" + RICH_INPUT_SLOT + "]";
}

/** Fire the paste event a file-manager copy produces: native File entries plus a filename text form. */
function pasteFilesInto(editor: HTMLElement, files: File[], textForm: string) {
  Object.defineProperty(editor, "isContentEditable", { configurable: true, value: true });
  editor.focus();

  const event = new Event("paste", { bubbles: true, cancelable: true }) as ClipboardEvent;

  Object.defineProperty(event, "clipboardData", {
    value: {
      getData: (type: string) => (type === "text" || type === "text/plain" ? textForm : ""),
      files: Object.assign(files, { item: (index: number) => files.at(index) ?? null }),
      items: files.map(file => ({ getAsFile: () => file, kind: "file", type: file.type }))
    }
  });

  act(() => {
    fireEvent(editor, event);
  });

  return event;
}

describe("a pasted OS file attaches instead of inserting text", () => {
  afterEach(() => {
    mainComposerScope.clear();
  });

  it("routes non-image file payloads through onAttachDroppedItems and keeps the text form out", () => {
    const seen: unknown[][] = [];
    const onAttachDroppedItems = vi.fn(async (...args: unknown[]) => {
      seen.push(args);
      return true;
    });
    const file = new File(["report contents"], "report.txt", { type: "text/plain" });

    const { container } = render(<Harness onAttachDroppedItems={onAttachDroppedItems} />);

    const editor = container.querySelector<HTMLElement>(editorSlot())!;

    const event = pasteFilesInto(editor, [file], "report.txt");

    expect(onAttachDroppedItems).toHaveBeenCalledTimes(1);
    expect(seen).toHaveLength(1);
    expect(seen[0]).toHaveLength(1);
    expect(event.defaultPrevented).toBe(true);
    expect(composerPlainText(editor)).not.toContain("report.txt");
  });

  it("leaves image pastes on the existing image-blob path", () => {
    const onAttachDroppedItems = vi.fn(async () => true);
    const image = new File(["bytes"], "photo.png", { type: "image/png" });

    const { container } = render(<Harness onAttachDroppedItems={onAttachDroppedItems} />);

    const editor = container.querySelector<HTMLElement>(editorSlot())!;

    pasteFilesInto(editor, [image], "");

    expect(onAttachDroppedItems).not.toHaveBeenCalled();
  });
});
