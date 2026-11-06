import type { CanonicalRpcMethods } from '../../../shared/src/gateway-contract.generated'

export type AccountUsageResponse = CanonicalRpcMethods['session.account_usage']['result']
export type AccountUsageSnapshot = NonNullable<AccountUsageResponse['snapshot']>
export type AccountUsageWindow = AccountUsageSnapshot['windows'][number]
export type AccountUsageRow = AccountUsageSnapshot['rows'][number]
