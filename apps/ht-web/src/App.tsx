import { useGateway } from "./gateway/useGateway";
import { MessageList } from "./components/MessageList";
import { Composer } from "./components/Composer";
import { SessionSidebar } from "./components/SessionSidebar";
import { ApprovalDialog, ClarifyDialog } from "./components/RequestDialogs";

const CONNECTION_LABEL: Record<string, string> = {
  idle: "Connecting…",
  connecting: "Connecting…",
  open: "Connected",
  closed: "Disconnected",
  error: "Connection failed",
};

export default function App() {
  const gw = useGateway();
  const { chat, connection, skin } = gw;
  const connected = connection === "open";
  const busy = chat.status === "working" || chat.status === "starting";

  return (
    <div className="ht-app">
      <SessionSidebar
        sessions={gw.sessions}
        activeId={gw.sessionId}
        agentName={skin.agentName}
        onNew={() => void gw.newSession()}
        onResume={(id) => void gw.resumeSession(id)}
      />

      <main className="ht-main">
        <header className="ht-header">
          <span className="ht-header__title">{skin.agentName}</span>
          <span className={`ht-status ht-status--${connection}`}>
            {chat.statusText || CONNECTION_LABEL[connection] || connection}
          </span>
        </header>

        <section className="ht-conversation">
          {chat.error && <div className="ht-error">{chat.error}</div>}
          <MessageList messages={chat.messages} />
        </section>

        {chat.clarify && (
          <ClarifyDialog clarify={chat.clarify} onRespond={(a) => void gw.respondClarify(a)} />
        )}
        {chat.approval && (
          <ApprovalDialog
            approval={chat.approval}
            onRespond={(c, all) => void gw.respondApproval(c, all)}
          />
        )}

        <Composer
          disabled={!connected || gw.sessionId === null}
          busy={busy}
          promptSymbol={skin.promptSymbol}
          onSubmit={(text) => void gw.submit(text)}
          onInterrupt={() => void gw.interrupt()}
        />
      </main>
    </div>
  );
}
