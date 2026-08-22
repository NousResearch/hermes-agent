# API 文档

## 响应格式说明

所有工具均返回 `content[].text`（自然语言文本）。部分工具（网约车类）额外返回 `structuredContent`（结构化数据）。

- 如果响应中存在 `structuredContent`，优先使用其中的字段做逻辑判断和字段提取
- 如果没有 `structuredContent`，则解析 `content[].text` 获取所需信息

## 函数签名

```
/**
   * 根据用户输入的起点终点坐标，规划骑行通勤方案
   *
   * @param destination 终点坐标，格式为：经度,纬度
   * @param need_geo? 是否需要返回途经的点序列，默认值为true
   * @param origin 起点坐标，格式为：经度,纬度
   */
  function maps_direction_bicycling(destination: string, need_geo?: boolean, origin: string);

  /**
   * 根据用户起终点经纬度坐标规划以小客车、轿车通勤出行的方案
   *
   * @param destination 终点坐标，格式为：经度,纬度
   * @param need_geo? 是否需要返回途经的点序列，默认值为true
   * @param origin 起点坐标，格式为：经度,纬度
   */
  function maps_direction_driving(destination: string, need_geo?: boolean, origin: string);

  /**
   * 根据用户起终点坐标，规划综合公交、地铁的通勤方案
   *
   * @param city 查询城市（**必须使用完整格式，如"北京市"而非"北京"**）
   * @param destination 终点坐标，格式为：经度,纬度
   * @param origin 起点坐标，格式为：经度,纬度
   */
  function maps_direction_transit(city: string, destination: string, origin: string);

  /**
   * 根据用户输入的起点终点坐标，规划步行通勤方案
   *
   * @param destination 终点坐标，格式为：经度,纬度
   * @param need_geo? 是否需要返回途经的点序列，默认值为true
   * @param origin 起点坐标，格式为：经度,纬度
   */
  function maps_direction_walking(destination: string, need_geo?: boolean, origin: string);

  /**
   * 根据用户传入关键词和位置坐标，搜索出周边的POI地点信息
   *
   * @param keywords 搜索关键词
   * @param location 位置坐标，格式为：经度,纬度
   * @param max_distance? 搜索半径，单位：米
   */
  function maps_place_around(keywords: string, location: string, max_distance?: string);

  /**
   * 将经纬度坐标转换为地址信息
   *
   * @param location 位置坐标，格式为：经度,纬度
   */
  function maps_regeocode(location: string);

  /**
   * 根据用户传入关键词和城市，搜索出相关的POI地点信息
   *
   * @param city 查询城市（**必须使用完整格式，如"北京市"而非"北京"**）
   * @param keywords 搜索关键词
   * @param location? 位置坐标，格式为：经度,纬度
   */
  function maps_textsearch(city: string, keywords: string, location?: string);

  /**
   * 取消打车订单
   *
   * @param order_id 订单ID，从订单创建或查询结果中获取
   * @param reason? 取消原因，可选参数。例如：不需要了、等待时间太长、临时有事等
   */
  function taxi_cancel_order(order_id: string, reason?: string);

  /**
   * 直接通过API创建打车订单，无需打开任何应用程序界面，系统自动完成整个发单流程
   *
   * @param caller_car_phone? 叫车人手机号，如果有就要传递，没有就不传
   * @param estimate_trace_id 预估流程ID，从预估结果中获取
   * @param product_category 车型品类标识，从预估结果中获取，传入多个车型时，用英文逗号分割，不要带空格
   * @returns structuredContent.orderId   订单ID，后续查询/取消订单使用
   * @returns structuredContent.status    订单初始状态（created）
   */
  function taxi_create_order(caller_car_phone?: string, estimate_trace_id: string, product_category: string);

  /**
   * 查看当前可用的网约车车型，请先获取对应地点的经纬度信息，如果有maps_textsearch的tool，优先使用maps_textsearch进行经纬度的获取。
   *
   * @param from_lat 出发纬度，必须从地图相关的工具获取，不能假设
   * @param from_lng 出发经度，必须从地图相关的工具获取，不能假设
   * @param from_name 出发地名称
   * @param to_lat 目的纬度，必须从地图相关的工具获取，不能假设
   * @param to_lng 目的经度，必须从地图相关的工具获取，不能假设
   * @param to_name 目的地名称
   * @returns structuredContent.traceId          预估流程ID，创建订单时必须传入
   * @returns structuredContent.items[]          可用车型列表
   * @returns structuredContent.items[].productName     车型名称
   * @returns structuredContent.items[].productCategory 车型品类代码，创建订单时传入
   * @returns structuredContent.items[].priceText       预估价格（元）
   */
  function taxi_estimate(from_lat: string, from_lng: string, from_name: string, to_lat: string, to_lng: string, to_name: string);

  /**
   * 根据起点、终点和车型生成打开移动应用或小程序的深度链接，用户点击后将跳转到相应的打车应用完成发单操作
   *
   * @param from_lat 出发纬度，必须从地图相关的工具获取，不能假设
   * @param from_lng 出发经度，必须从地图相关的工具获取，不能假设
   * @param product_category? 车型品类标识列表，从预估结果中获取，支持多个车型，仅当用户明确指定某个或某些品类时才传递此参数，格式为英文逗号,分割
   * @param to_lat 目的地纬度，必须从地图相关的工具获取，不能假设
   * @param to_lng 目的经度，必须从地图相关的工具获取，不能假设
   */
  function taxi_generate_ride_app_link(from_lat: string, from_lng: string, product_category?: string, to_lat: string, to_lng: string);

  /**
   * 获取打车订单对应司机的实时位置经纬度
   *
   * @param order_id 打车订单ID
   */
  function taxi_get_driver_location(order_id: string);

  /**
   * 查询打车订单的状态和信息，如司机联系方式、车牌号、预估到达时间
   *
   * ⚠️ 重要：此函数单次调用仅返回当前状态。
   * 单次调用返回当前状态，详见 SKILL.md `### 3.6 查询订单`。
   *
   * @param order_id? 订单ID，从创建订单结果中获取，如果有就要传递，如果没有，会查询当前账号下未完成的订单
   * @returns structuredContent.statusCode  订单状态码（见下表）
   * @returns structuredContent.statusText  状态文本描述
   * @returns structuredContent.driver      司机信息（name/phone/carModel/carPlate），接单后可用
   * @returns structuredContent.map.distanceKm  距离（公里），行程中可用
   * @returns structuredContent.map.eta         预计剩余时间（分钟），行程中可用
   * @returns structuredContent.map.phase       当前阶段：to_pickup（前往上车点）| to_dropoff（前往终点）
   *
   * 订单状态码：
   *   0  匹配中（非终态）
   *   1  司机已接单（非终态）
   *   2  司机已到达（非终态，里程碑通知）
   *   3  未知状态（终态）
   *   4  行程开始（非终态，里程碑通知）
   *   5  订单完成（终态）✅
   *   6  订单已被系统取消（终态）✅
   *   7  订单已被取消（终态）✅
   *   8  未知状态（终态）✅
   *   9  未知状态（终态）✅
   *   10 未知状态（终态）✅
   *   11 客服关闭订单（终态）✅
   *   12 未能完成服务（终态）✅
   */
  function taxi_query_order(order_id?: string);
```

Examples:

```bash
# 设置 MCP_URL 变量
MCP_URL="https://mcp.didichuxing.com/mcp-servers?key=$DIDI_MCP_KEY"

# 地址解析（city 建议使用完整行政区名称）
mcporter call "$MCP_URL" maps_textsearch --args '{"keywords":"望京SOHO","city":"北京市"}'

# 价格预估（注意：所有参数值必须加引号，使用字符串格式）
mcporter call "$MCP_URL" taxi_estimate --args '{"from_lat":"39.9","from_lng":"116.4","from_name":"望京SOHO","to_lat":"39.9","to_lng":"116.4","to_name":"国贸"}'
```
