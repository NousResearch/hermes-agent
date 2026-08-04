import { defineConfig } from "@kubb/core";
import { pluginOas } from "@kubb/plugin-oas";
import { pluginTs } from "@kubb/plugin-ts";
import { pluginZod } from "@kubb/plugin-zod";

export default defineConfig({
  root: ".",
  input: {
    // 团队记忆扩展版契约：在 offload.yaml 13 个接口基础上叠加可选 IdFields
    // (team_id / agent_id / user_id / task_id)，用于服务化模式的身份隔离。
    // 旧客户端不传 IdFields 时按 offload.yaml 原语义工作。
    path: "./docs/team-api-仅memory.yaml",
  },
  output: {
    path: "./src/gateway/generated",
    clean: true,
    barrelType: false,
  },
  plugins: [
    pluginOas({
      generators: [],
    }),
    pluginTs({
      output: {
        path: "./types.ts",    // single file
        barrelType: false,
      },
    }),
    pluginZod({
      output: {
        path: "./schemas.ts",  // single file
        barrelType: false,
      },
      typed: true,
      importPath: "zod",
    }),
  ],
});
