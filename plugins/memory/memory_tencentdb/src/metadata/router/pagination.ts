/** Re-export for router/schemas（实现见 metadata/pagination.ts）。 */
export {
  paginationInputSchema,
  resolvePagination,
  toPaginationParams,
  wrapPaginated,
  paginateArray,
  formatListResult,
  isPaginatedResult,
  unwrapListItems,
  DEFAULT_PAGINATION,
} from "../pagination.js";
