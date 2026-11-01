// MobileComputeServer.swift - Implements the HTTP server for Mobile Compute API.
//
// Endpoints:
//   GET /health          -> {"status": "ok", "device": "iPhone 16e"}
//   GET /capabilities    -> {"device": "iPhone 16e", "compute": ["echo"], "coreml": false, "llm": false}
//   POST /compute        -> POST /compute with task_id, task_type, payload
//   GET /tasks           -> GET /tasks (list all tasks)
//   GET /tasks/{id}      -> GET /tasks/{id} (get task status)

import Foundation
import NIOCore
import NIOPosix
import NIOHTTP1

/// Configuration for the Mobile Compute Server
struct MobileComputeConfig: Codable {
    let enabled: Bool
    let host: String
    let port: UInt16
    let tls: TLSConfig?
    
    struct TLSConfig: Codable {
        let enabled: Bool
        let certPath: String?
        let keyPath: String?
        let caPath: String?
        let requireClientCert: Bool?
    }
    
    static func load() -> MobileComputeConfig {
        // For now, return default config
        // In production, load from a config file
        return MobileComputeConfig(
            enabled: true,
            host: "0.0.0.0",
            port: 8765,
            tls: nil
        )
    }
}

/// A compute task in the queue
struct MobileComputeTask: Codable {
    let task_id: UUID
    let task_type: String
    let payload: [String: String]
    var status: String = "completed"
    var result: String? = nil
}

/// JSON response helper
struct JSONResponse: Codable {
    let status: String
    let device: String
}

/// Capabilities response
struct CapabilitiesResponse: Codable {
    let device: String
    let compute: [String]
    let coreml: Bool
    let llm: Bool
}

/// Compute request
struct ComputeRequest: Codable {
    let task_id: String?
    let task_type: String
    let payload: [String: String]
}

/// Compute response
struct ComputeResponse: Codable {
    let success: Bool
    let task_id: String
    let status: String
    let result: String?
    let error: String?
}

/// Tasks list response
struct TasksResponse: Codable {
    let tasks: [MobileComputeTask]
}

/// Task status response
struct TaskStatusResponse: Codable {
    let task_id: String
    let task_type: String
    let status: String
    let payload: [String: String]
    let result: String?
    let error: String?
    let created_at: Double?
    let started_at: Double?
    let completed_at: Double?
}

/// HTTP channel handler for processing requests
final class MobileComputeHandler: ChannelInboundHandler {
    typealias InboundIn = HTTPServerRequestPart
    typealias OutboundOut = HTTPServerResponsePart
    
    private var currentRequest: HTTPRequestHead?
    private var bodyBuffer: ByteBuffer?
    private let server: MobileComputeServer
    
    init(server: MobileComputeServer) {
        self.server = server
    }
    
    func channelRead(context: ChannelHandlerContext, data: NIOAny) {
        let requestPart = unwrapInboundIn(data)
        
        switch requestPart {
        case .head(let head):
            currentRequest = head
            bodyBuffer = context.channel.allocator.buffer(capacity: 0)
            
        case .body(var buffer):
            if var body = bodyBuffer {
                body.writeBuffer(&buffer)
                bodyBuffer = body
            }
            
        case .end:
            guard let head = currentRequest, let body = bodyBuffer else { return }
            handleRequest(context: context, head: head, body: body)
            currentRequest = nil
            bodyBuffer = nil
        }
    }
    
    private func handleRequest(context: ChannelHandlerContext, head: HTTPRequestHead, body: ByteBuffer) {
        let path = head.uri
        let method = head.method
        
        // Route to appropriate handler
        switch (method, path) {
        case (.GET, "/health"):
            handleHealth(context: context)
        case (.GET, "/capabilities"):
            handleCapabilities(context: context)
        case (.POST, "/compute"):
            handleCompute(context: context, body: body)
        case (.GET, "/tasks"):
            handleTasks(context: context)
        case (.GET, let p) where p.hasPrefix("/tasks/"):
            let taskId = String(p.dropFirst("/tasks/".count))
            handleTaskStatus(context: context, taskId: taskId)
        default:
            sendResponse(context: context, status: .notFound, body: "{\"error\": \"Not found\"}")
        }
    }
    
    private func handleHealth(context: ChannelHandlerContext) {
        let response = JSONResponse(status: "ok", device: "iPhone 16e")
        let json = try! JSONEncoder().encode(response)
        sendResponse(context: context, status: .ok, body: json)
    }
    
    private func handleCapabilities(context: ChannelHandlerContext) {
        let response = CapabilitiesResponse(
            device: "iPhone 16e",
            compute: ["echo"],
            coreml: false,
            llm: false
        )
        let json = try! JSONEncoder().encode(response)
        sendResponse(context: context, status: .ok, body: json)
    }
    
    private func handleCompute(context: ChannelHandlerContext, body: ByteBuffer) {
        // A ByteBuffer is not a Foundation.Data, so it cannot be handed to
        // JSONDecoder directly. `readableBytesView` is the zero-copy slice of
        // readable bytes that the buffer API guarantees to cover exactly
        // `readableBytes` bytes; Data(_:) copies those bytes out.
        let data = Data(body.readableBytesView)

        // A truncated or absent body is a client error, not a decode failure.
        guard !data.isEmpty else {
            sendComputeError(context: context, message: "Empty request body")
            return
        }

        // Parse request body
        let request: ComputeRequest
        do {
            let decoder = JSONDecoder()
            request = try decoder.decode(ComputeRequest.self, from: data)
        } catch {
            sendComputeError(context: context, message: "Invalid JSON: \(error)")
            return
        }
        
        // Process task (echo for now)
        let taskId = request.task_id ?? UUID().uuidString
        let result = request.payload["text"] ?? "echo: \(request.payload)"
        
        let task = MobileComputeTask(
            task_id: UUID(uuidString: taskId) ?? UUID(),
            task_type: request.task_type,
            payload: request.payload,
            status: "completed",
            result: result
        )
        server.record(task: task)
        
        let response = ComputeResponse(
            success: true,
            task_id: taskId,
            status: "completed",
            result: result,
            error: nil
        )
        
        let json = try! JSONEncoder().encode(response)
        sendResponse(context: context, status: .ok, body: json)
    }
    
    private func handleTasks(context: ChannelHandlerContext) {
        let response = TasksResponse(tasks: server.tasksSnapshot)
        let json = try! JSONEncoder().encode(response)
        sendResponse(context: context, status: .ok, body: json)
    }
    
    private func handleTaskStatus(context: ChannelHandlerContext, taskId: String) {
        if let task = server.tasksSnapshot.first(where: { $0.task_id.uuidString == taskId }) {
            let response = TaskStatusResponse(
                task_id: task.task_id.uuidString,
                task_type: task.task_type,
                status: task.status,
                payload: task.payload,
                result: task.result,
                error: nil,
                created_at: nil,
                started_at: nil,
                completed_at: nil
            )
            let json = try! JSONEncoder().encode(response)
            sendResponse(context: context, status: .ok, body: json)
        } else {
            sendResponse(context: context, status: .notFound, body: "{\"error\": \"Task not found\"}")
        }
    }
    
    /// Sends a failed ComputeResponse with the given client-side error.
    ///
    /// Factored out so the empty-body and decode-failure paths build the same
    /// response shape the original single error branch produced.
    private func sendComputeError(context: ChannelHandlerContext, message: String) {
        let response = ComputeResponse(
            success: false,
            task_id: UUID().uuidString,
            status: "failed",
            result: nil,
            error: message
        )
        let json = try! JSONEncoder().encode(response)
        sendResponse(context: context, status: .badRequest, body: json)
    }

    private func sendResponse(context: ChannelHandlerContext, status: HTTPResponseStatus, body: Data) {
        var buffer = context.channel.allocator.buffer(capacity: body.count)
        buffer.writeBytes(body)
        
        let responseHead = HTTPResponseHead(version: .http1_1, status: status, headers: [
            "Content-Type": "application/json",
            "Content-Length": "\(body.count)"
        ])
        
        context.write(self.wrapOutboundOut(.head(responseHead)), promise: nil)
        context.write(self.wrapOutboundOut(.body(.byteBuffer(buffer))), promise: nil)
        context.writeAndFlush(self.wrapOutboundOut(.end(nil)), promise: nil)
    }
    
    private func sendResponse(context: ChannelHandlerContext, status: HTTPResponseStatus, body: String) {
        let data = body.data(using: .utf8)!
        sendResponse(context: context, status: status, body: data)
    }
}

/// Main server class
final class MobileComputeServer: @unchecked Sendable {
    let config: MobileComputeConfig
    private let lock = NSLock()
    private var tasks: [MobileComputeTask] = []
    private var group: EventLoopGroup?
    private var bootstrap: ServerBootstrap?
    private var channel: Channel?
    
    init(config: MobileComputeConfig) {
        self.config = config
    }
    
    /// Snapshot of completed tasks, safe to read from any NIO event loop.
    var tasksSnapshot: [MobileComputeTask] {
        lock.lock()
        defer { lock.unlock() }
        return tasks
    }

    /// Records a completed task.
    func record(task: MobileComputeTask) {
        lock.lock()
        defer { lock.unlock() }
        tasks.append(task)
        // Keep the in-memory list bounded; this is an echo-only Phase 0 node.
        if tasks.count > 128 {
            tasks.removeFirst(tasks.count - 128)
        }
    }

    /// Starts the HTTP server
    func start() async throws {
        group = MultiThreadedEventLoopGroup(numberOfThreads: System.coreCount)
        
        let handler = MobileComputeHandler(server: self)
        
        bootstrap = ServerBootstrap(group: group!)
            .serverChannelOption(ChannelOptions.backlog, value: 256)
            .serverChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
            .childChannelInitializer { channel in
                channel.pipeline.configureHTTPServerPipeline(withErrorHandling: true).flatMap {
                    channel.pipeline.addHandler(handler)
                }
            }
            .childChannelOption(ChannelOptions.socketOption(.so_reuseaddr), value: 1)
            .childChannelOption(ChannelOptions.maxMessagesPerRead, value: 16)
            .childChannelOption(ChannelOptions.recvAllocator, value: AdaptiveRecvByteBufferAllocator())
        
        let address = try SocketAddress.makeAddressResolvingHost(config.host, port: Int(config.port))
        channel = try await bootstrap!.bind(to: address).get()
        
        NSLog("Mobile Compute server listening on \(config.host):\(config.port)")
    }
    
    /// Stops the HTTP server and releases the EventLoopGroup.
    func stop() async throws {
        if let channel = self.channel {
            try? await channel.close().get()
            self.channel = nil
        }
        if let group = self.group {
            try? await group.shutdownGracefully()
            self.group = nil
        }
        self.bootstrap = nil
    }
}
