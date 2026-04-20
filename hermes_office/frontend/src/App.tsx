import React, { useEffect, useState } from "react";
import { useStore } from "./state";
import { OfficeSocket } from "./ws";
import { OfficeCanvas } from "./components/OfficeCanvas";
import { TopBar } from "./components/TopBar";
import { Sidebar } from "./components/Sidebar";
import { TaskComposer } from "./components/TaskComposer";
import { HireWizard } from "./components/HireWizard";
import { DepartmentManager } from "./components/DepartmentManager";
import { EmployeeEditor } from "./components/EmployeeEditor";
import { ActivityRail } from "./components/ActivityRail";
import { OnboardingOverlay } from "./components/OnboardingOverlay";
import { Toasts, useToasts } from "./components/Toasts";
import { t } from "./i18n";

type Modal = "hire" | "department" | "employee" | null;

export default function App() {
  const ready = useStore((s) => s.ready);
  const initialise = useStore((s) => s.initialise);
  const refreshEmployees = useStore((s) => s.refreshEmployees);
  const refreshDepartments = useStore((s) => s.refreshDepartments);
  const refreshCapacity = useStore((s) => s.refreshCapacity);
  const pushActivity = useStore((s) => s.pushActivity);
  const applyState = useStore((s) => s.applyEmployeeStateChange);
  const employees = useStore((s) => s.employees);
  const departments = useStore((s) => s.departments);
  const selectedEmpId = useStore((s) => s.selectedEmpId);
  const selectEmp = useStore((s) => s.selectEmp);

  const [modal, setModal] = useState<Modal>(null);
  const [, force] = useState(0);
  const toasts = useToasts();

  useEffect(() => {
    initialise().catch((e) => {
      console.error(e);
      toasts.push({ tone: "error", title: "Backend unreachable", body: String(e) });
    });
    const onLang = () => force((n) => n + 1);
    window.addEventListener("hermes_office_lang_change", onLang);
    return () => window.removeEventListener("hermes_office_lang_change", onLang);
  }, [initialise]);

  useEffect(() => {
    const sock = new OfficeSocket();
    const off = sock.subscribe((msg) => {
      if (msg.kind === "hello") return;
      pushActivity(msg);
      if (msg.kind === "state_change") {
        const to = (msg.meta as Record<string, string>)?.to;
        if (to) applyState(msg.employee_id, to as any);
      }
      if (msg.kind === "state_change" && (msg.meta as any)?.to === "resting") {
        // employee finished — refresh capacity in case employee count changed
      }
    });
    return () => {
      off();
      sock.close();
    };
  }, [pushActivity, applyState]);

  useEffect(() => {
    const id = setInterval(() => {
      refreshCapacity().catch(() => undefined);
    }, 30_000);
    return () => clearInterval(id);
  }, [refreshCapacity]);

  if (!ready) {
    return (
      <div className="h-screen w-screen flex items-center justify-center text-ink">
        <div className="text-center">
          <div className="animate-pulse text-2xl mb-2 font-display font-semibold">{t("appTitle")}</div>
          <div className="opacity-70 text-sm">loading…</div>
        </div>
      </div>
    );
  }

  const empty = employees.length === 0 && departments.length === 0;

  const onHired = async () => {
    setModal(null);
    await Promise.all([refreshEmployees(), refreshDepartments(), refreshCapacity()]);
    toasts.push({ tone: "ok", title: "Hired!" });
  };

  const onDeptCreated = async () => {
    setModal(null);
    await refreshDepartments();
    toasts.push({ tone: "ok", title: "Department created" });
  };

  const onEmpUpdated = async () => {
    setModal(null);
    await refreshEmployees();
    toasts.push({ tone: "ok", title: "Saved" });
  };

  return (
    <div className="h-screen w-screen flex flex-col text-ink overflow-hidden">
      <TopBar onHire={() => setModal("hire")} onAddDept={() => setModal("department")} />
      <div className="flex-1 flex min-h-0">
        <Sidebar onPickEmployee={(id) => { selectEmp(id); setModal("employee"); }} />
        <main className="flex-1 relative min-w-0 flex flex-col">
          <div className="flex-1 relative">
            <OfficeCanvas
              onPickEmployee={(id) => { selectEmp(id); setModal("employee"); }}
            />
            {empty && <OnboardingOverlay onHire={() => setModal("hire")} />}
          </div>
          <TaskComposer />
        </main>
        <ActivityRail />
      </div>

      {modal === "hire" && (
        <HireWizard onClose={() => setModal(null)} onHired={onHired} />
      )}
      {modal === "department" && (
        <DepartmentManager onClose={() => setModal(null)} onCreated={onDeptCreated} />
      )}
      {modal === "employee" && selectedEmpId && (
        <EmployeeEditor
          empId={selectedEmpId}
          onClose={() => { selectEmp(null); setModal(null); }}
          onSaved={onEmpUpdated}
          onDeleted={async () => {
            selectEmp(null);
            setModal(null);
            await refreshEmployees();
            toasts.push({ tone: "ok", title: "Removed" });
          }}
        />
      )}
      <Toasts ctrl={toasts} />
    </div>
  );
}
