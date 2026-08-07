/**
 * 元数据 list 接口统一分页（offset 风格）。
 *
 * 入参：limit/offset 可选；未传时默认 limit=20、offset=0。
 * 出参：始终为 PaginatedResult 信封。
 */
import { z } from "zod";
import type {
  ListPage,
  PaginatedResult,
  PaginationInput,
  PaginationParams,
} from "./types.js";

export const DEFAULT_PAGINATION: PaginationParams = { limit: 20, offset: 0 };

/** 请求体中的可选分页字段（merge 到各 list schema，不单独 default 以免 Zod merge 行为异常）。 */
export const paginationInputSchema = z.object({
  limit: z.coerce.number().int().min(1).max(100).optional(),
  offset: z.coerce.number().int().min(0).optional(),
});

const paginationResolvedSchema = z.object({
  limit: z.coerce.number().int().min(1).max(100).default(20),
  offset: z.coerce.number().int().min(0).default(0),
});

/** 解析分页入参（未传 limit/offset 时使用默认 20/0）。 */
export function resolvePagination(input?: PaginationInput): PaginationParams {
  return paginationResolvedSchema.parse(input ?? {});
}

/** @deprecated 使用 resolvePagination */
export function toPaginationParams(input?: PaginationInput): PaginationParams {
  return resolvePagination(input);
}

export function wrapPaginated<T>(
  items: T[],
  total: number,
  pagination: PaginationParams,
): PaginatedResult<T> {
  return {
    items,
    total,
    limit: pagination.limit,
    offset: pagination.offset,
  };
}

/** 判断是否为分页信封（Control/SDK 解析时兼容历史裸数组）。 */
export function isPaginatedResult<T>(value: unknown): value is PaginatedResult<T> {
  return (
    value !== null
    && typeof value === "object"
    && !Array.isArray(value)
    && "items" in value
    && Array.isArray((value as PaginatedResult<T>).items)
    && "total" in value
  );
}

/** 解包 list 响应（兼容 PaginatedResult 与历史裸数组 T[]）。 */
export function unwrapListItems<T>(value: T[] | PaginatedResult<T>): T[] {
  return isPaginatedResult(value) ? value.items : value;
}

/** list API 统一出参。 */
export function formatListResult<T>(
  page: ListPage<T>,
  pagination: PaginationParams,
): PaginatedResult<T> {
  return wrapPaginated(page.items, page.total, pagination);
}

/** 对已物化数组做 offset 分页（listAccessibleAssets 等权限聚合场景）。 */
export function paginateArray<T>(items: T[], pagination: PaginationParams): PaginatedResult<T> {
  const total = items.length;
  const slice = items.slice(pagination.offset, pagination.offset + pagination.limit);
  return wrapPaginated(slice, total, pagination);
}
