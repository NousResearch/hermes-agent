const js = require("@eslint/js");
const globals = require("globals");

module.exports = [
  {
    files: ["src/**/*.js"],
    ...js.configs.recommended,
    languageOptions: {
      ecmaVersion: 2022,
      sourceType: "module",
      globals: globals.browser,
    },
  },
  {
    files: ["src/app.js"],
    rules: {
      "no-empty": "off",
      "no-unused-vars": "off",
      "no-useless-escape": "off",
    },
  },
];
