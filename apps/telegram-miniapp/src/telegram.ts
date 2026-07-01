export type TelegramThemeParams = {
  bg_color?: string;
  text_color?: string;
  hint_color?: string;
  link_color?: string;
  button_color?: string;
  button_text_color?: string;
  secondary_bg_color?: string;
  header_bg_color?: string;
  accent_text_color?: string;
  section_bg_color?: string;
  section_header_text_color?: string;
  subtitle_text_color?: string;
  destructive_text_color?: string;
};

export type TelegramWebAppUser = {
  id: number;
  first_name?: string;
  last_name?: string;
  username?: string;
  language_code?: string;
  is_premium?: boolean;
};

type TelegramButton = {
  show: () => void;
  hide: () => void;
  onClick: (callback: () => void) => void;
  offClick: (callback: () => void) => void;
};

type TelegramMainButton = TelegramButton & {
  setText: (text: string) => void;
  enable: () => void;
  disable: () => void;
};

type TelegramWebApp = {
  initData?: string;
  initDataUnsafe?: {
    user?: TelegramWebAppUser;
    auth_date?: number;
    hash?: string;
  };
  themeParams?: TelegramThemeParams;
  colorScheme?: "light" | "dark";
  platform?: string;
  viewportHeight?: number;
  viewportStableHeight?: number;
  BackButton?: TelegramButton;
  MainButton?: TelegramMainButton;
  HapticFeedback?: {
    impactOccurred?: (style: "light" | "medium" | "heavy" | "rigid" | "soft") => void;
    notificationOccurred?: (type: "error" | "success" | "warning") => void;
    selectionChanged?: () => void;
  };
  ready?: () => void;
  expand?: () => void;
};

declare global {
  interface Window {
    Telegram?: {
      WebApp?: TelegramWebApp;
    };
  }
}

export type TelegramRuntime = {
  isTelegram: boolean;
  initData: string;
  user?: TelegramWebAppUser;
  theme: TelegramThemeParams;
  colorScheme: "light" | "dark";
  platform: string;
};

export function prepareTelegramViewport(): void {
  const webApp = window.Telegram?.WebApp;
  webApp?.ready?.();
  webApp?.expand?.();
}

export function configureTelegramBackButton(isVisible: boolean, onBack: () => void): () => void {
  const backButton = window.Telegram?.WebApp?.BackButton;
  if (!backButton) return () => undefined;

  if (isVisible) {
    backButton.show();
    backButton.onClick(onBack);
  } else {
    backButton.hide();
  }

  return () => {
    backButton.offClick(onBack);
    backButton.hide();
  };
}

export function configureTelegramMainButton(text: string, onClick: () => void): () => void {
  const mainButton = window.Telegram?.WebApp?.MainButton;
  if (!mainButton) return () => undefined;

  mainButton.setText(text);
  mainButton.disable();
  mainButton.show();
  mainButton.onClick(onClick);

  return () => {
    mainButton.offClick(onClick);
    mainButton.hide();
  };
}

export function triggerTelegramRefreshHaptic(type: "start" | "success" | "warning"): void {
  const haptic = window.Telegram?.WebApp?.HapticFeedback;
  if (!haptic) return;

  if (type === "start") {
    haptic.selectionChanged?.();
    return;
  }

  haptic.notificationOccurred?.(type === "success" ? "success" : "warning");
}

export function getTelegramRuntime(): TelegramRuntime {
  const webApp = window.Telegram?.WebApp;
  if (!webApp) {
    return {
      isTelegram: false,
      initData: "",
      theme: {},
      colorScheme: "dark",
      platform: "browser-mock",
    };
  }

  return {
    isTelegram: true,
    initData: webApp.initData ?? "",
    user: webApp.initDataUnsafe?.user,
    theme: webApp.themeParams ?? {},
    colorScheme: webApp.colorScheme ?? "dark",
    platform: webApp.platform ?? "telegram",
  };
}