import { create } from "zustand";
import { api } from "./api";
import type { ActivityEvent, CapacityReport, Department, Employee, Preset, SkillInfo, Task, ToolsetInfo } from "./types";

interface State {
  ready: boolean;
  departments: Department[];
  employees: Employee[];
  tasks: Task[];
  presets: Preset[];
  skills: SkillInfo[];
  toolsets: ToolsetInfo[];
  capacity: CapacityReport | null;
  selectedDeptId: string | null;
  selectedEmpId: string | null;
  activityByEmp: Record<string, ActivityEvent[]>;
  initialise: () => Promise<void>;
  refreshDepartments: () => Promise<void>;
  refreshEmployees: () => Promise<void>;
  refreshCapacity: () => Promise<void>;
  selectDept: (id: string | null) => void;
  selectEmp: (id: string | null) => void;
  pushActivity: (e: ActivityEvent) => void;
  applyEmployeeStateChange: (emp_id: string, activity: Employee["activity"]) => void;
}

const ACTIVITY_LIMIT = 50;

export const useStore = create<State>((set, get) => ({
  ready: false,
  departments: [],
  employees: [],
  tasks: [],
  presets: [],
  skills: [],
  toolsets: [],
  capacity: null,
  selectedDeptId: null,
  selectedEmpId: null,
  activityByEmp: {},

  async initialise() {
    const [departments, employees, presets, skills, toolsets, capacity] = await Promise.all([
      api.listDepartments(),
      api.listEmployees(),
      api.presets(),
      api.skills(),
      api.toolsets(),
      api.capacity().catch(() => null),
    ]);
    set({
      departments,
      employees,
      presets,
      skills,
      toolsets,
      capacity,
      selectedDeptId: departments[0]?.id ?? null,
      ready: true,
    });
  },

  async refreshDepartments() {
    set({ departments: await api.listDepartments() });
  },
  async refreshEmployees() {
    set({ employees: await api.listEmployees() });
  },
  async refreshCapacity() {
    set({ capacity: await api.capacity().catch(() => null) });
  },

  selectDept(id) {
    set({ selectedDeptId: id });
  },
  selectEmp(id) {
    set({ selectedEmpId: id });
  },

  pushActivity(e) {
    const { activityByEmp } = get();
    const list = activityByEmp[e.employee_id] ?? [];
    const next = [...list, e].slice(-ACTIVITY_LIMIT);
    set({ activityByEmp: { ...activityByEmp, [e.employee_id]: next } });
  },

  applyEmployeeStateChange(emp_id, activity) {
    const { employees } = get();
    set({
      employees: employees.map((e) => (e.id === emp_id ? { ...e, activity } : e)),
    });
  },
}));
