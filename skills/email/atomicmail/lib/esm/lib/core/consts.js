import { tryReadSharedJson } from "./shared-assets.js";
const SHARED_CONSTS = tryReadSharedJson("consts.json") ?? {
    DEFAULT_POW_SCRYPT_SALT_HEX: "0b980734412c292d6549110276b604ab1dea4883bd460d77d1b984adf8bca083",
    DEFAULT_AUTH_URL: "https://auth.atomicmail.ai",
    DEFAULT_API_URL: "https://api.atomicmail.ai",
    ONE_SEC_MS: 1000,
    ONE_MIN_MS: 1000 * 60,
    ONE_HOUR_MS: 1000 * 60 * 60,
    ONE_DAY_MS: 1000 * 60 * 60 * 24,
    ONE_MONTH_MS: 1000 * 60 * 60 * 24 * 30,
    ONE_YEAR_MS: 1000 * 60 * 60 * 24 * 365,
};
/**
 * Fixed proof-of-work scrypt salt. The auth-service passes this string (UTF-8
 * bytes of the hex text, not decoded binary) to `scrypt` as the `salt`
 * argument; all PoW clients must use the same value.
 */
export const DEFAULT_POW_SCRYPT_SALT_HEX = SHARED_CONSTS.DEFAULT_POW_SCRYPT_SALT_HEX;
/** Production auth-service base URL when unset in env and credentials.json. */
export const DEFAULT_AUTH_URL = SHARED_CONSTS.DEFAULT_AUTH_URL;
/** Production JMAP / API base URL when unset in env and credentials.json. */
export const DEFAULT_API_URL = SHARED_CONSTS.DEFAULT_API_URL;
export const ONE_SEC_MS = SHARED_CONSTS.ONE_SEC_MS;
export const ONE_MIN_MS = SHARED_CONSTS.ONE_MIN_MS;
export const ONE_HOUR_MS = SHARED_CONSTS.ONE_HOUR_MS;
export const ONE_DAY_MS = SHARED_CONSTS.ONE_DAY_MS;
export const ONE_MONTH_MS = SHARED_CONSTS.ONE_MONTH_MS;
export const ONE_YEAR_MS = SHARED_CONSTS.ONE_YEAR_MS;
