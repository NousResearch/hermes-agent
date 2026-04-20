// Tiny i18n with EN + zh-CN bundles.

type Lang = "en" | "zh-CN";

const dict = {
  en: {
    appTitle: "Hermes Office",
    hireBtn: "Hire",
    addDeptBtn: "Department",
    capacity: "Capacity",
    runtime: "Runtime",
    runtimeSim: "Demo",
    runtimeReal: "Live",
    work: "Work",
    talk: "Talk",
    rest: "Rest",
    learn: "Learn",
    composerPlaceholder: "Type a task… use @dept or @name to direct it",
    send: "Send",
    step1Title: "Pick a body",
    step2Title: "Pick a job — or describe it",
    step3Title: "Confirm",
    confirmHire: "Hire",
    cancel: "Cancel",
    next: "Next",
    back: "Back",
    describePlaceholder: "Describe in your own words (any language)…",
    suggestSkills: "Auto-pick skills",
    name: "Name",
    role: "Role",
    model: "Model",
    skills: "Skills",
    toolsets: "Toolsets",
    persona: "Persona",
    activity: "Activity",
    danger: "Danger",
    delete: "Delete",
    cliCopy: "Copy CLI command",
    deptName: "Department name",
    deptMission: "Mission",
    deptColor: "Color",
    deptCreate: "Create department",
    employees: "employees",
    queued: "queued",
    running: "running",
    done: "done",
    failed: "failed",
    welcomeTitle: "Welcome to your AI office",
    welcomeBody: "Click the green + Hire button to add your first employee.",
  },
  "zh-CN": {
    appTitle: "Hermes 办公室",
    hireBtn: "招人",
    addDeptBtn: "新部门",
    capacity: "产能",
    runtime: "运行模式",
    runtimeSim: "演示",
    runtimeReal: "真跑",
    work: "工作区",
    talk: "交流区",
    rest: "休息区",
    learn: "学习区",
    composerPlaceholder: "输入任务,可用 @部门 或 @姓名 指派",
    send: "派发",
    step1Title: "选个形象",
    step2Title: "选个工种,或自己描述",
    step3Title: "确认入职",
    confirmHire: "入职!",
    cancel: "取消",
    next: "下一步",
    back: "上一步",
    describePlaceholder: "用你自己的话描述他要做什么 (中文/English 都可以)",
    suggestSkills: "自动挑技能",
    name: "姓名",
    role: "工种",
    model: "模型",
    skills: "技能",
    toolsets: "工具",
    persona: "人设",
    activity: "活动",
    danger: "危险操作",
    delete: "删除",
    cliCopy: "复制命令行",
    deptName: "部门名",
    deptMission: "部门使命",
    deptColor: "代表色",
    deptCreate: "创建部门",
    employees: "位员工",
    queued: "排队中",
    running: "工作中",
    done: "完成",
    failed: "失败",
    welcomeTitle: "欢迎来到你的 AI 办公室",
    welcomeBody: "点击右下角绿色 + 招人 按钮,添加第一位员工。",
  },
};

let currentLang: Lang = (() => {
  const q = new URLSearchParams(location.search).get("lang");
  if (q === "zh" || q === "zh-CN") return "zh-CN";
  if (q === "en") return "en";
  const saved = localStorage.getItem("hermes_office_lang");
  if (saved === "zh-CN" || saved === "en") return saved;
  return navigator.language?.startsWith("zh") ? "zh-CN" : "en";
})();

export function setLang(lang: Lang) {
  currentLang = lang;
  localStorage.setItem("hermes_office_lang", lang);
  // Force a re-render by dispatching an event the App listens for.
  window.dispatchEvent(new CustomEvent("hermes_office_lang_change"));
}

export function getLang(): Lang {
  return currentLang;
}

export function t(key: keyof typeof dict["en"]): string {
  return (dict[currentLang] as Record<string, string>)[key] ?? (dict.en as Record<string, string>)[key] ?? String(key);
}
