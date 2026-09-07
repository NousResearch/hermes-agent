# ECS Troubleshooting

基于真实测试暴露的常见错误和诊断步骤。

## 错误码映射

| 错误                                    | 根因                            | 修复                                                                                                     |
| --------------------------------------- | ------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `Ecs.0005` flavor-image 不匹配          | 镜像虚拟化类型与规格不兼容      | 检查 image `__support_amd`/`__support_xen` 与 flavor `os_extra_specs`。BareMetal 镜像不兼容标准 ECS 规格 |
| `[USE_ERROR]不正确的参数:imagetype`     | 参数名错误                      | KooCLI 使用双下划线前缀：`--__imagetype=gold`（不是 `--imagetype`）                                      |
| `[USE_ERROR]不正确的参数:chargingMode`  | 参数路径错误                    | 计费模式参数路径：`--server.extendparam.chargingMode=postPaid`。不传默认按需付费                         |
| `[USE_ERROR]不正确的参数:__support_amd` | 参数无效                        | `IMS ListImages` 某些版本不支持此过滤。去掉该参数，使用 client-side 过滤                                 |
| Cannot SSH                              | 安全组未开放22端口 或 未绑定EIP | 1) 添加入方向规则 tcp 22。2) `hcloud EIP BindPublicIp`                                                   |
| Flavor unavailable                      | 区域不支持该规格                | `hcloud ECS ListFlavors --cli-region=<r>` 先查。不硬编码 s6/m6 等规格名                                  |
| Insufficient resources                  | AZ 库存不足                     | 换规格、换 AZ、或等待资源释放                                                                            |
| AuthFailure                             | AK/SK 过期或无效                | `npx huaweicloud-devkit auth init` 重新配置统一凭据                                                      |

## 创建失败诊断流程

1. 运行 `hcloud ECS ListFlavors` — 确认规格在目标区域可用
2. 运行 `hcloud IMS ListImages --__imagetype=gold --limit=20` — 确认可用的公共镜像
3. 交叉检查：flavor `os_extra_specs` 中的虚拟化类型 vs image 的 `__support_*` 属性
4. 运行 `hcloud VPC ListVpcs` / `ListSubnets` — 确认 VPC 和子网存在
5. 运行 `hcloud ECS NovaListAvailabilityZones` — 确认 AZ 可用
6. 创建后：`hcloud ECS ListServersDetails --server_id=<id>` 检查状态
7. 状态 `BUILD` → 等待。`ERROR` → 查看 job_id 详情

## 镜像查询正确姿势

```bash
# 正确：双下划线前缀
hcloud IMS ListImages --cli-region=<r> --__imagetype=gold --__isregistered=true --limit=20

# 按需过滤（如果参数支持）
# --virtual_env_type=FusionCompute  排除 BareMetal
# 不支持的参数不要传，用 client-side 过滤
```

## 密钥对 vs 密码

| 方式                       | 安全  | 建议                                               |
| -------------------------- | ----- | -------------------------------------------------- |
| `--server.key_name=<name>` | ✅ 高 | **推荐**。先 `hcloud DEW CreateKeypair` 创建密钥对 |
| `--server.adminPass=<pw>`  | ❌ 低 | 密码明文进入 shell 历史。仅测试用                  |

密码要求：8-26 字符，含大写 + 小写 + 数字 + 特殊字符。

## 删除注意事项

- `--delete_publicip` 默认 false — EIP 不随实例删除，继续计费
- `--delete_volume` 默认 false — 系统盘不随实例删除
- 数据盘默认随实例删除（与系统盘行为相反）
