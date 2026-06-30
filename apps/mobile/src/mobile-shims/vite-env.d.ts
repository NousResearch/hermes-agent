/// <reference types="react" />

declare module '*.css' {
  const value: string;
  export default value;
}

declare module '*.jsonl?raw' {
  const value: string;
  export default value;
}

declare module '*?raw' {
  const value: string;
  export default value;
}

interface ImportMetaEnv {
  readonly MODE: string;
  readonly DEV?: boolean;
  readonly PROD?: boolean;
  readonly [key: string]: string | boolean | undefined;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
