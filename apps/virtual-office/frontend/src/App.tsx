import { Link, Route, Routes, useLocation } from "react-router-dom";
import AgentsPanel from "./pages/AgentsPanel";
import ConsoleLogs from "./pages/ConsoleLogs";
import HandoffLogs from "./pages/HandoffLogs";
import MainOffice from "./pages/MainOffice";
import Settings from "./pages/Settings";
import TaskBoard from "./pages/TaskBoard";
import TradeRoom from "./pages/TradeRoom";

const navItems = [
  { to: "/", label: "Main Office" },
  { to: "/task-board", label: "Task Board" },
  { to: "/trade-room", label: "Trade Room" },
  { to: "/handoff-logs", label: "Handoff Logs" },
  { to: "/console-logs", label: "Console Logs" },
  { to: "/agents", label: "Agents Panel" },
  { to: "/settings", label: "Settings" },
];

export default function App() {
  const location = useLocation();

  return (
    <div style={{ minHeight: "100vh", color: "#e2e8f0" }}>
      <header
        style={{
          position: "sticky",
          top: 0,
          zIndex: 20,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 24,
          padding: "18px 28px",
          backdropFilter: "blur(16px)",
          background: "rgba(8, 15, 28, 0.75)",
          borderBottom: "1px solid rgba(148, 163, 184, 0.12)",
        }}
      >
        <div>
          <div style={{ fontWeight: 800, letterSpacing: "0.08em", textTransform: "uppercase" }}>Hermes Virtual Office</div>
          <div style={{ color: "#94a3b8", fontSize: 13 }}>Control Layer and Operator Workflow</div>
        </div>
        <nav style={{ display: "flex", gap: 10, flexWrap: "wrap", justifyContent: "flex-end" }}>
          {navItems.map((item) => {
            const active = location.pathname === item.to;
            return (
              <Link
                key={item.to}
                to={item.to}
                style={{
                  padding: "10px 14px",
                  borderRadius: 999,
                  background: active ? "rgba(52, 211, 153, 0.18)" : "rgba(15, 23, 42, 0.75)",
                  border: `1px solid ${active ? "rgba(52, 211, 153, 0.35)" : "rgba(148, 163, 184, 0.12)"}`,
                  color: active ? "#d1fae5" : "#cbd5e1",
                  fontWeight: 700,
                }}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
      </header>

      <main style={{ padding: 28 }}>
        <Routes>
          <Route path="/" element={<MainOffice />} />
          <Route path="/task-board" element={<TaskBoard />} />
          <Route path="/trade-room" element={<TradeRoom />} />
          <Route path="/handoff-logs" element={<HandoffLogs />} />
          <Route path="/console-logs" element={<ConsoleLogs />} />
          <Route path="/agents" element={<AgentsPanel />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </main>
    </div>
  );
}
